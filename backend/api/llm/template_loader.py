# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Jinja2 template loader for model-specific prompt formatting.

This module provides a secure, cached template loading system for LLM prompts.
Each model can have its own template that defines system prompt formatting,
tool call syntax, and reasoning structure.

Security: Uses Jinja2's SandboxedEnvironment to prevent arbitrary code execution.
Performance: Templates are cached after first load.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment

logger = logging.getLogger(__name__)

# Template directory relative to this file
TEMPLATES_DIR = Path(__file__).parent / "templates"


class TemplateLoader:
    """Secure template loader with caching for LLM prompts.
    
    Uses Jinja2's SandboxedEnvironment for security and caches
    compiled templates for performance.
    
    Example:
        loader = TemplateLoader()
        prompt = loader.render("nemotron", {
            "system_prompt": "You are Aevom...",
            "tools": [...],
            "supports_reasoning": True,
        })
    """
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize the template loader.
        
        Args:
            templates_dir: Optional custom templates directory.
                          Defaults to ./templates/ relative to this file.
        """
        self._templates_dir = templates_dir or TEMPLATES_DIR
        
        # Ensure templates directory exists
        if not self._templates_dir.exists():
            logger.warning(f"Templates directory does not exist: {self._templates_dir}")
            self._templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Use SandboxedEnvironment for security - prevents arbitrary code execution
        self._env = SandboxedEnvironment(
            loader=FileSystemLoader(str(self._templates_dir)),
            # Trim whitespace around blocks for cleaner output
            trim_blocks=True,
            lstrip_blocks=True,
            # Keep trailing newline for readability
            keep_trailing_newline=True,
        )
        
        logger.info(f"TemplateLoader initialized with templates from: {self._templates_dir}")
    
    @lru_cache(maxsize=32)
    def _get_template(self, template_name: str):
        """Get a compiled template by name (cached).
        
        Args:
            template_name: Name of the template (without .jinja2 extension).
            
        Returns:
            Compiled Jinja2 template.
            
        Raises:
            TemplateNotFound: If template file doesn't exist.
        """
        template_file = f"{template_name}.jinja2"
        return self._env.get_template(template_file)
    
    def render(
        self,
        template_name: str,
        context: dict[str, Any],
        fallback_to_base: bool = True,
    ) -> str:
        """Render a template with the given context.
        
        Args:
            template_name: Name of the template (e.g., "nemotron", "qwen3").
            context: Dictionary of variables to pass to the template.
            fallback_to_base: If True, falls back to "base" template if
                            the specified template is not found.
        
        Returns:
            Rendered template string.
            
        Raises:
            TemplateNotFound: If template not found and fallback_to_base is False.
        """
        try:
            template = self._get_template(template_name)
            rendered = template.render(**context)
            logger.debug(f"Rendered template '{template_name}' successfully")
            return rendered
        except TemplateNotFound:
            if fallback_to_base and template_name != "base":
                logger.warning(
                    f"Template '{template_name}' not found, falling back to 'base'"
                )
                return self.render("base", context, fallback_to_base=False)
            raise
    
    def has_template(self, template_name: str) -> bool:
        """Check if a template exists.
        
        Args:
            template_name: Name of the template to check.
            
        Returns:
            True if template exists, False otherwise.
        """
        template_file = self._templates_dir / f"{template_name}.jinja2"
        return template_file.exists()
    
    def list_templates(self) -> list[str]:
        """List all available templates.
        
        Returns:
            List of template names (without .jinja2 extension).
        """
        if not self._templates_dir.exists():
            return []
        return [
            p.stem for p in self._templates_dir.glob("*.jinja2")
        ]
    
    def clear_cache(self) -> None:
        """Clear the template cache.
        
        Useful for development when templates are being modified.
        """
        self._get_template.cache_clear()
        logger.info("Template cache cleared")


# Global template loader instance
_loader: Optional[TemplateLoader] = None


def get_template_loader() -> TemplateLoader:
    """Get the global template loader instance.
    
    Returns:
        Global TemplateLoader instance.
    """
    global _loader
    if _loader is None:
        _loader = TemplateLoader()
    return _loader


def render_template(
    template_name: str,
    context: dict[str, Any],
    fallback_to_base: bool = True,
) -> str:
    """Convenience function to render a template.
    
    Args:
        template_name: Name of the template.
        context: Variables for the template.
        fallback_to_base: Fall back to base template if not found.
        
    Returns:
        Rendered template string.
    """
    return get_template_loader().render(template_name, context, fallback_to_base)
