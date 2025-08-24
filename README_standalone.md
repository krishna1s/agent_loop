# Standalone Website Exploration API

A single-file Python API that provides autonomous website exploration with real-time screenshot capture using Semantic Kernel and Playwright MCP.

## Features

- 🤖 **Autonomous Agent**: Uses Semantic Kernel ChatCompletionAgent for intelligent website exploration
- 🎭 **Browser Automation**: Integrates with Playwright MCP server for real browser control
- 📸 **Real-time Screenshots**: Captures and stores screenshots at every step
- 🔄 **Background Processing**: Runs exploration tasks in background threads
- 🌐 **REST API**: Clean REST endpoints for task management and screenshot access
- 📊 **Progress Tracking**: Real-time status updates and progress monitoring

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_standalone.txt
```

### 2. Install Playwright MCP Server

```bash
npm install -g @playwright/mcp
```

### 3. Environment Variables

Set up your Azure OpenAI credentials:

```bash
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_KEY="your-api-key"
```

## Usage

### 1. Start the API Server

```bash
python standalone_exploration_api.py
```

The API will be available at: `http://localhost:8001`
API documentation: `http://localhost:8001/docs`

### 2. Start Website Exploration

**POST** `/start_exploration`

```json
{
  "url": "https://example.com",
  "task_description": "Explore the website and generate a comprehensive test plan"
}
```

Response:
```json
{
  "task_id": "12345678-1234-5678-9012-123456789012",
  "status": "started",
  "message": "Exploration task started for https://example.com"
}
```

### 3. Monitor Task Progress

**GET** `/task_status/{task_id}`

Response:
```json
{
  "task_id": "12345678-1234-5678-9012-123456789012",
  "status": "running",
  "progress": "Exploring navigation menu",
  "latest_action": "Clicked on About page",
  "completed": false
}
```

### 4. Get Latest Screenshot

**GET** `/screenshot`

Response:
```json
{
  "screenshot_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "timestamp": "2025-08-05T10:30:45.123456",
  "status": "available"
}
```

### 5. List Active Tasks

**GET** `/active_tasks`

Response:
```json
{
  "active_tasks": [
    {
      "task_id": "12345678-1234-5678-9012-123456789012",
      "url": "https://example.com",
      "status": "running",
      "progress": "Exploring forms",
      "created_at": "2025-08-05T10:25:00.123456"
    }
  ],
  "total_tasks": 1
}
```

### 6. Stop a Task

**DELETE** `/task/{task_id}`

Response:
```json
{
  "message": "Task 12345678-1234-5678-9012-123456789012 stopped and removed"
}
```

## Example Usage Script

Run the example script to see the API in action:

```bash
python example_api_usage.py
```

This script will:
1. Check API health
2. Start website exploration
3. Monitor progress in real-time
4. Save screenshots as they're captured
5. Display final results

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/start_exploration` | Start new exploration task |
| GET | `/task_status/{task_id}` | Get task status and progress |
| GET | `/screenshot` | Get latest screenshot as base64 |
| GET | `/active_tasks` | List all active tasks |
| DELETE | `/task/{task_id}` | Stop and remove task |
| GET | `/health` | API health check |

## Agent Behavior

The exploration agent will:

1. **Navigate** to the specified URL
2. **Capture screenshots** after every action
3. **Discover** all interactive elements on the page
4. **Systematically explore** user flows:
   - Authentication flows (login, signup)
   - Navigation flows (menus, links)
   - Form submission flows
   - Interactive features (dropdowns, modals)
5. **Generate test plans** with step-by-step documentation
6. **Continue exploration** until all discoverable functionality is documented

## Screenshot Storage

- Screenshots are automatically captured after every browser action
- Stored as base64 encoded PNG images
- Available via `/screenshot` endpoint
- Also saved locally in `screenshots/` directory
- MCP server saves originals to `/tmp/playwright-mcp-output/`

## Task Management

- Tasks run in background threads
- Multiple concurrent tasks supported
- Real-time status updates
- Automatic cleanup on completion
- Error handling and recovery

## Troubleshooting

### Common Issues

1. **API won't start**: Check Azure OpenAI environment variables
2. **No screenshots**: Ensure Playwright MCP server is installed (`npm install -g @playwright/mcp`)
3. **Browser errors**: Check if Edge browser is available
4. **Task stuck**: Use DELETE endpoint to stop and restart

### Debug Mode

The API includes extensive debug logging. Check console output for detailed information about:
- MCP plugin initialization
- Browser actions and responses
- Screenshot capture process
- Task execution progress

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │ Semantic Kernel  │    │ Playwright MCP  │
│   REST API      │◄──►│ ChatCompletion   │◄──►│ Browser Control │
│                 │    │ Agent            │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Background     │    │  Custom Tools    │    │  Screenshot     │
│  Task Threads   │    │  & Functions     │    │  Capture        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Files

- `standalone_exploration_api.py` - Main API server (single file solution)
- `requirements_standalone.txt` - Python dependencies
- `example_api_usage.py` - Example usage script
- `README.md` - This documentation

## License

This is a demonstration tool for website exploration and test plan generation.
