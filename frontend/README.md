# Website Exploration Agent React UI

This is the React frontend for the Website Exploration Agent that provides a split-screen interface:
- **Left Panel**: Chat interface for user interactions
- **Right Panel**: Live browser screenshots and analysis

## Features

- 🌐 **Real-time Browser View**: Live screenshots of website exploration
- 💬 **Interactive Chat**: Communicate with the AutoGen agent
- ⚡ **WebSocket Connection**: Real-time updates and responses
- 🎯 **Quick Actions**: One-click navigation and snapshot tools
- 📊 **Element Analysis**: Detailed element discovery information
- 🔄 **Auto-reconnection**: Automatic WebSocket reconnection on disconnect

## Setup

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Set environment variables** (optional):
   ```bash
   # Create .env file
   REACT_APP_WS_URL=ws://localhost:8001/ws
   REACT_APP_API_URL=http://localhost:8001/api
   ```

3. **Start development server**:
   ```bash
   npm start
   ```

The app will open at `http://localhost:3000` and connect to the FastAPI backend at `http://localhost:8001`.

## Architecture

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   React UI      │◄───────────────►│  FastAPI Backend│
│                 │                 │                 │
│ ┌─────────────┐ │                 │ ┌─────────────┐ │
│ │ Chat Panel  │ │                 │ │   AutoGen   │ │
│ │             │ │                 │ │    Agent    │ │
│ └─────────────┘ │                 │ └─────────────┘ │
│ ┌─────────────┐ │                 │ ┌─────────────┐ │
│ │Browser Panel│ │                 │ │ Playwright  │ │
│ │             │ │                 │ │   Browser   │ │
│ └─────────────┘ │                 │ └─────────────┘ │
└─────────────────┘                 └─────────────────┘
```

## Usage

1. **Start Exploration**: Type exploration requests like:
   - "Navigate to https://example.com"
   - "Take a snapshot of the current page"
   - "Click on the login button"
   - "Find all interactive elements"

2. **View Results**: Browser screenshots appear in real-time on the right panel

3. **Interactive Iteration**: Continue the conversation to refine exploration

## API Communication

The frontend communicates with the backend via:
- **WebSocket** (`/ws`): Real-time messaging and updates
- **REST API** (`/api/*`): Direct API calls when needed

Message types:
- `exploration_request`: User requests for website exploration
- `browser_action`: Direct browser actions (navigate, click, type)
- `exploration_response`: Agent responses with screenshots
- `status`: Status updates during processing
- `error`: Error messages and handling
