import logging
import os
from PIL import Image

import numpy as np

import fiftyone.core.models as fom
import fiftyone.utils.torch as fout
from fiftyone.utils.torch import ClassifierOutputProcessor

from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class NomicEmbedConfig(fout.TorchImageModelConfig):
    """
    Config class for Nomic Embed Multimodal 7B.
    
    Nomic Embed Multimodal is a 7B parameter vision-language model that generates
    3584-dimensional embeddings for both images and text in a shared vector space,
    enabling visual document retrieval and zero-shot classification.
    
    Args:
        model_path (str): HuggingFace model ID. Default: "nomic-ai/nomic-embed-multimodal-7b"
        
        text_prompt (str): Optional baseline text prompt for classification. Default: ""
        
        use_flash_attention (bool): Whether to use Flash Attention 2 if available. Default: True
    """

    def __init__(self, d):
        """Initialize the configuration.

        Args:
            d: A dictionary containing the configuration parameters
        """
        # Processor handles preprocessing, so use raw inputs
        if "raw_inputs" not in d:
            d["raw_inputs"] = True
        
        # Only set up output processor if classes provided (for classification)
        if "classes" in d and d["classes"] is not None and len(d["classes"]) > 0:
            if "output_processor_cls" not in d:
                d["output_processor_cls"] = "fiftyone.utils.torch.ClassifierOutputProcessor"
        
        super().__init__(d)
        
        # Nomic-specific configuration
        self.model_path = self.parse_string(
            d, "model_path", default="nomic-ai/nomic-embed-multimodal-7b"
        )
        self.text_prompt = self.parse_string(d, "text_prompt", default="")


class NomicEmbedModel(fout.TorchImageModel, fom.PromptMixin):
    """
    Nomic Embed Multimodal 7B model for document understanding and retrieval.
    
    This model generates 3584-dimensional single-vector embeddings for both images
    and text, enabling:
    - High-quality similarity search and retrieval
    - Zero-shot classification
    - Embeddings visualization
    
    Based on Qwen2.5-VL architecture with 7B parameters, this model provides
    state-of-the-art performance for visual document understanding.
    
    The model extends TorchImageModel for image processing and PromptMixin for text embedding.
    """
    
    def __init__(self, config):
        """Initialize the model.
        
        Args:
            config: A NomicEmbedMultimodalConfig instance containing model parameters
        """
        # Initialize parent classes
        super().__init__(config)
        
        # Storage for cached data
        self._text_features = None  # Cached text features for classification
        self._last_computed_embeddings = None  # Last computed 3584-dim embeddings
        

    @property
    def has_embeddings(self):
        """Whether this instance can generate embeddings."""
        return True

    @property
    def can_embed_prompts(self):
        """Whether this instance can embed text prompts."""
        return True
    
    @property
    def classes(self):
        """The list of class labels for the model."""
        return self._classes

    @classes.setter
    def classes(self, value):
        """Set new classes and invalidate cached text features."""
        self._classes = value
        self._text_features = None  # Invalidate cache
        
        # Rebuild output processor if classes are provided
        if value is not None and len(value) > 0:
            self._output_processor = ClassifierOutputProcessor(classes=value)
        else:
            self._output_processor = None
    
    @property
    def text_prompt(self):
        """The text prompt prefix for classification."""
        return self.config.text_prompt

    @text_prompt.setter  
    def text_prompt(self, value):
        """Set new text prompt and invalidate cached text features."""
        self.config.text_prompt = value
        self._text_features = None  # Invalidate cache
    
    def _load_model(self, config):
        """Load Nomic Embed Multimodal model and processor from HuggingFace.
        
        Args:
            config: NomicEmbedMultimodalConfig instance containing model parameters

        Returns:
            model: The loaded model
        """

        
        logger.info(f"Loading Nomic Embed Multimodal model from {config.model_path}")
        
        model_kwargs = {
            "device_map": self.device,
        }

        # Set optimizations based on device capabilities
        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(self._device)
            
            # Use bfloat16 for Ampere or newer GPUs (capability >= 8.0)
            if capability[0] >= 8:
                model_kwargs["dtype"] = torch.bfloat16
            else:
                model_kwargs["dtype"] = torch.float16

        # Enable flash attention if available
        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load processor
        self.processor = BiQwen2_5_Processor.from_pretrained(config.model_path)
        
        # Load model
        self.model = BiQwen2_5.from_pretrained(
            config.model_path, 
            **model_kwargs
            )
        
        self.model.eval()
    
        
        return self.model

    def _prepare_images_for_processor(self, imgs):
        """Convert images to PIL format (processor's expected input).
        
        Args:
            imgs: List of images (PIL, numpy arrays, or tensors)
            
        Returns:
            List of PIL Images
        """
        pil_images = []
        
        for img in imgs:
            if isinstance(img, Image.Image):
                # Already PIL Image
                pil_images.append(img)
            elif isinstance(img, torch.Tensor):
                # Tensor (CHW) → PIL Image
                img_np = img.permute(1, 2, 0).cpu().numpy()
                if img_np.dtype != np.uint8:
                    # Assume normalized [0, 1] or [-1, 1]
                    if img_np.min() < 0:
                        img_np = ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    else:
                        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            elif isinstance(img, np.ndarray):
                # Numpy array (HWC) → PIL Image
                if img.dtype != np.uint8:
                    # Assume normalized [0, 1] or [0, 255]
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                pil_images.append(Image.fromarray(img))
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
        
        return pil_images

    def _get_text_features(self):
        """Get or compute text features for classification.
        
        Creates embeddings for each class by combining text_prompt with class names.
        
        Returns:
            torch.Tensor: Text embeddings with shape (num_classes, 3584)
        """
        if self._text_features is None:
            # Create prompts for each class
            prompts = [
                "%s %s" % (self.config.text_prompt, c) for c in self.classes
            ]
            # Compute and cache text features
            self._text_features = self._embed_prompts_internal(prompts)
        
        return self._text_features
    
    def _embed_prompts_internal(self, prompts):
        """Embed text prompts using processor and model.
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            torch.Tensor: Text embeddings with shape (batch, 3584)
        """
        # Process queries through processor (note: process_queries, not process_texts)
        query_inputs = self.processor.process_queries(prompts)
        
        # Move to device
        query_inputs = {k: v.to(self._device) for k, v in query_inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model(**query_inputs)
        
        return embeddings

    def embed_prompt(self, prompt):
        """Embed a single text prompt to 3584-dim vector for retrieval.
        
        Args:
            prompt: Text prompt to embed
            
        Returns:
            numpy array: 3584-dim embedding vector
        """
        embeddings = self._embed_prompts_internal([prompt])
        result = embeddings[0].cpu().numpy()
        return result

    def embed_prompts(self, prompts):
        """Embed multiple text prompts to 3584-dim vectors for retrieval.
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            numpy array: 3584-dim embeddings with shape (batch, 3584)
        """
        embeddings = self._embed_prompts_internal(prompts)
        result = embeddings.cpu().numpy()
        return result

    def embed_images(self, imgs):
        """Embed images to 3584-dim vectors for retrieval/similarity search.
        
        Args:
            imgs: List of images (PIL, numpy arrays, or tensors)
            
        Returns:
            numpy array: 3584-dim embeddings with shape (batch, 3584)
        """
        # Convert to PIL images
        pil_images = self._prepare_images_for_processor(imgs)
        
        # Process images through processor
        image_inputs = self.processor.process_images(pil_images)
        
        # Move to device
        image_inputs = {k: v.to(self._device) for k, v in image_inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model(**image_inputs)
            
            # Cache for get_embeddings()
            self._last_computed_embeddings = embeddings
        
        return embeddings.cpu().numpy()
    
    def embed(self, img):
        """Embed a single image.
        
        Args:
            img: PIL image to embed
            
        Returns:
            numpy array: 3584-dim embedding
        """
        embeddings = self.embed_images([img])
        return embeddings[0]

    def embed_all(self, imgs):
        """Embed a batch of images.
        
        Args:
            imgs: List of images to embed
            
        Returns:
            numpy array: 3584-dim embeddings
        """
        return self.embed_images(imgs)
    
    def get_embeddings(self):
        """Get the last computed 3584-dim embeddings.
        
        Returns:
            numpy array: The last computed embeddings with shape (batch, 3584)
        """
        if not self.has_embeddings:
            raise ValueError("This model instance does not expose embeddings")
        
        if self._last_computed_embeddings is None:
            raise ValueError("No embeddings have been computed yet")
        
        result = self._last_computed_embeddings.cpu().numpy()
        return result

    def _get_class_logits(self, text_features, image_features):
        """Calculate similarity scores using cosine similarity.
        
        Uses the processor's built-in score method for efficient computation.
        
        Args:
            text_features: Text embeddings (num_classes, 3584)
            image_features: Image embeddings (num_images, 3584)
            
        Returns:
            tuple: (logits_per_image, logits_per_text)
                - logits_per_image: shape (num_images, num_classes)
                - logits_per_text: shape (num_classes, num_images)
        """
        with torch.no_grad():
            # Convert to list of 1D tensors (processor.score expects List[Tensor])
            # Using torch.unbind like in native usage
            text_list = list(torch.unbind(text_features))  # List of (3584,) tensors
            image_list = list(torch.unbind(image_features))  # List of (3584,) tensors
            
            # Use processor's built-in scoring (cosine similarity)
            logits_per_text = self.processor.score(
                text_list, 
                image_list,
                device=self._device
            )  # Returns: (num_classes, num_images)
            
            logits_per_image = logits_per_text.t()
            
            return logits_per_image, logits_per_text

    def _predict_all(self, imgs):
        """Run zero-shot classification on a batch of images.
        
        Uses cosine similarity between image and class text embeddings.
        
        Args:
            imgs: List of images to classify
            
        Returns:
            Classification predictions processed by output processor
        """
        # Check if classification is supported
        if self.classes is None or len(self.classes) == 0:
            raise ValueError(
                "Cannot perform classification without classes. "
                "Set classes when loading: foz.load_zoo_model(..., classes=['class1', 'class2'])"
            )
        
        if self._output_processor is None:
            raise ValueError(
                "No output processor configured for classification."
            )
        
        # Get image embeddings
        image_embeddings = self.embed_images(imgs)
        image_features = torch.from_numpy(image_embeddings).to(self._device)
        
        # Get cached text features for classes
        text_features = self._get_text_features()
        
        # Calculate cosine similarity
        output, _ = self._get_class_logits(text_features, image_features)
        
        # Get frame size for output processor
        if isinstance(imgs[0], torch.Tensor):
            height, width = imgs[0].size()[-2:]
        elif hasattr(imgs[0], 'size'):  # PIL Image
            width, height = imgs[0].size
        else:
            height, width = imgs[0].shape[:2]  # numpy array
        
        frame_size = (width, height)
        
        if self.has_logits:
            self._output_processor.store_logits = self.store_logits
        
        return self._output_processor(
            output, 
            frame_size, 
            confidence_thresh=self.config.confidence_thresh
        )