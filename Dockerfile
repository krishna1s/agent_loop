FROM ubuntu:22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (no nodejs/npm here)
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-distutils python3-pip \
    curl git build-essential \
    wget ca-certificates gnupg \
    fonts-liberation fonts-unifont fonts-ubuntu \
    libasound2 libatk-bridge2.0-0 libatk1.0-0 libcups2 libdbus-1-3 \
    libdrm2 libgbm1 libgtk-3-0 libnspr4 libnss3 libxcomposite1 \
    libxdamage1 libxrandr2 xdg-utils libu2f-udev libvulkan1 \
    libxshmfence1 libxfixes3 libxext6 libxi6 libxrender1 libxtst6 \
    libnss3-tools libxss1 libxcb1 libx11-6 libfontconfig1 \
    supervisor nginx \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS and npm via NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update && apt-get install -y nodejs \
    && npm config set fund false \
    && npm config set audit false \
    && npm config set update-notifier false \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && ln -sf /usr/bin/pip3 /usr/bin/pip

# Install Python dependencies
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Install Playwright and browsers (Chromium + deps) and also Chrome channel
RUN pip install playwright && playwright install --with-deps chromium
RUN npm i -g @playwright/mcp@latest 

    # Set environment for MCP Playwright server to be local SSE
ENV PLAYWRIGHT_MCP_SSE_URL="http://127.0.0.1:8931/sse"

# Create launch script for Chromium with a CDP (Chrome DevTools Protocol) endpoint
COPY launch_chromium.sh /app/launch_chromium.sh
COPY wait_for_cdp_and_start_mcp.sh /app/wait_for_cdp_and_start_mcp.sh
RUN chmod +x /app/launch_chromium.sh /app/wait_for_cdp_and_start_mcp.sh

# Screenshot loop script (Playwright connects over CDP to existing browser)
COPY screenshot_loop.py /app/screenshot_loop.py

# Copy app code and configs
COPY . .
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY nginx.conf /etc/nginx/nginx.conf


# Expose nginx and remote browser ports
EXPOSE 80 9222

# Start all services
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]