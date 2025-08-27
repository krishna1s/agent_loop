FROM ubuntu:22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-distutils python3-pip \
    curl git build-essential \
    wget ca-certificates \
    fonts-liberation fonts-unifont fonts-ubuntu \
    libasound2 libatk-bridge2.0-0 libatk1.0-0 libcups2 libdbus-1-3 \
    libdrm2 libgbm1 libgtk-3-0 libnspr4 libnss3 libxcomposite1 \
    libxdamage1 libxrandr2 xdg-utils libu2f-udev libvulkan1 \
    supervisor nginx \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && ln -sf /usr/bin/pip3 /usr/bin/pip

# Install Python dependencies
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Install Playwright and browsers
RUN pip install playwright && playwright install --with-deps chromium

# Copy app code and configs
COPY . .
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY nginx.conf /etc/nginx/nginx.conf

# Expose only nginx port
EXPOSE 80

# Start all services
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]