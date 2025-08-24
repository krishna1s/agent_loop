#!/usr/bin/env python3
"""
Playwright MCP Configuration for Website Exploration Agent
Configures Playwright MCP tools with headed mode for visual browser automation
"""

import asyncio
import json
from typing import Any, Dict, Optional
import os
import subprocess
import tempfile
from pathlib import Path

class PlaywrightMCPManager:
    """Manages Playwright MCP server with headed mode configuration"""
    
    def __init__(self, headless: bool = False):
        """
        Initialize Playwright MCP Manager
        
        Args:
            headless: False for headed mode (visible browser), True for headless
        """
        self.headless = headless
        self.mcp_server_process = None
        self.config_file = None
        
    async def start_mcp_server(self) -> Dict[str, Any]:
        """Start the Playwright MCP server with headed mode configuration"""
        try:
            # Create temporary config file for MCP server
            config = {
                "headless": self.headless,
                "devtools": not self.headless,  # Enable devtools in headed mode
                "slowMo": 500 if not self.headless else 0,  # Slow down actions in headed mode
                "viewport": {"width": 1280, "height": 720},
                "browser": "chromium",  # Use Chromium for best compatibility
                "args": [] if self.headless else ["--start-maximized"]
            }
            
            # Write config to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config, f, indent=2)
                self.config_file = f.name
            
            return {
                "success": True,
                "config": config,
                "config_file": self.config_file,
                "message": f"MCP server configured for {'headed' if not self.headless else 'headless'} mode"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to configure Playwright MCP server"
            }
    
    async def stop_mcp_server(self):
        """Stop the MCP server and clean up"""
        try:
            if self.mcp_server_process:
                self.mcp_server_process.terminate()
                await asyncio.sleep(1)
                if self.mcp_server_process.poll() is None:
                    self.mcp_server_process.kill()
                    
            if self.config_file and os.path.exists(self.config_file):
                os.unlink(self.config_file)
                
        except Exception as e:
            print(f"Error stopping MCP server: {e}")
    
    def get_browser_config(self) -> Dict[str, Any]:
        """Get browser configuration for Playwright"""
        return {
            "headless": self.headless,
            "devtools": not self.headless,
            "slowMo": 500 if not self.headless else 0,
            "viewport": {"width": 1280, "height": 720},
            "args": [] if self.headless else [
                "--start-maximized",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--allow-running-insecure-content"
            ]
        }

# Global MCP manager instance
playwright_mcp = PlaywrightMCPManager(headless=False)  # Set to False for headed mode
