# Deployment Guide for Gradio App Service

## Initial Setup

### 1. System Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3 python3-pip python3-venv git supervisor nginx

# Install poetry (optional, if you're using it)
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Create Application Directory

```bash
# Create application directory
sudo mkdir /opt/gradio-app
sudo chown $USER:$USER /opt/gradio-app

# Clone your repository
git clone https://github.com/your-username/your-repo.git /opt/gradio-app
cd /opt/gradio-app

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Create Systemd Service

Create a service file:

```bash
sudo nano /etc/systemd/system/gradio-app.service
```

Add the following content:

```ini
[Unit]
Description=Gradio App Service
After=network.target

[Service]
User=your-username
Group=your-username
WorkingDirectory=/opt/gradio-app
Environment="PATH=/opt/gradio-app/venv/bin"
ExecStart=/opt/gradio-app/venv/bin/python app.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### 4. Configure Nginx as Reverse Proxy

Create Nginx configuration:

```bash
sudo nano /etc/nginx/sites-available/gradio-app
```

Add the following content:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/gradio-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 5. Start the Service

```bash
# Enable and start the service
sudo systemctl enable gradio-app
sudo systemctl start gradio-app

# Check status
sudo systemctl status gradio-app
```

## Updating the Application

### 1. Create Update Script

Create `/opt/gradio-app/update.sh`:

```bash
#!/bin/bash

# Go to app directory
cd /opt/gradio-app

# Pull latest changes
git pull

# Activate virtual environment
source venv/bin/activate

# Update dependencies
pip install -r requirements.txt

# Restart service
sudo systemctl restart gradio-app

# Check service status
sudo systemctl status gradio-app
```

Make it executable:

```bash
chmod +x /opt/gradio-app/update.sh
```

### 2. Update Process

To update the application:

```bash
# Run update script
sudo /opt/gradio-app/update.sh
```

## Monitoring and Maintenance

### View Logs

```bash
# View service logs
sudo journalctl -u gradio-app -f

# View nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Common Commands

```bash
# Restart service
sudo systemctl restart gradio-app

# Stop service
sudo systemctl stop gradio-app

# Start service
sudo systemctl start gradio-app

# Check service status
sudo systemctl status gradio-app
```

## SSL Configuration (Optional)

Install Certbot and obtain SSL certificate:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Environment Variables

If your app uses environment variables:

1. Create environment file:
```bash
sudo nano /etc/systemd/system/gradio-app.service.d/override.conf
```

2. Add environment variables:
```ini
[Service]
Environment="URL=your_url_value"
Environment="REPLICATE_API_TOKEN=your_token"
```

3. Reload systemd:
```bash
sudo systemctl daemon-reload
sudo systemctl restart gradio-app
```