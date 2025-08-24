# Website Exploration Agent with React UI

This project combines Microsoft AutoGen's multi-agent framework with a modern React frontend to create an intelligent website exploration agent that systematically explores websites and generates comprehensive test plans using real browser automation.

## 🌟 Features

- **🎨 Split-Screen React UI**: Modern interface with chat on left, live browser view on right
- **🌐 Real-time Browser Automation**: Microsoft Edge integration with live screenshots
- **💬 Interactive Chat Interface**: WebSocket-based real-time communication
- **📊 Element Discovery**: Automatic detection and analysis of interactive elements
- **🔄 Iterative Exploration**: Continuous discovery and documentation of user flows
- **⚡ Quick Actions**: One-click navigation and snapshot tools
- **🎯 Test Plan Generation**: Comprehensive markdown documentation output
- **🔌 WebSocket Communication**: Real-time updates and responses
- **🔄 Auto-reconnection**: Robust connection management

## 🚀 Quick Start

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd testing-autogen
   ```

2. **Set Environment Variables**:
   ```bash
   export AZURE_OPENAI_ENDPOINT="your-azure-openai-endpoint"
   export AZURE_OPENAI_API_KEY="your-api-key"
   export AZURE_OPENAI_API_VERSION="2024-06-01"
   export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"
   ```

3. **Start the full stack**:
   ```bash
   ./start.sh
   ```

4. **Open Browser**: Navigate to `http://localhost:3000` to start exploring!

## 🏗️ Architecture

```
┌─────────────────────┐    WebSocket     ┌─────────────────────┐    ┌─────────────────────┐
│    React Frontend   │◄───────────────►│   FastAPI Backend   │◄──►│    AutoGen Agent    │
│                     │                  │                     │    │                     │
│ ┌─────────────────┐ │                  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │
│ │   Chat Panel    │ │                  │ │  WebSocket API  │ │    │ │ System Message  │ │
│ │   (Left Side)   │ │                  │ │                 │ │    │ │   + Tools       │ │
│ └─────────────────┘ │                  │ └─────────────────┘ │    │ └─────────────────┘ │
│ ┌─────────────────┐ │                  │ ┌─────────────────┐ │    └─────────────────────┘
│ │ Browser Panel   │ │                  │ │   REST API      │ │              │
│ │  (Right Side)   │ │                  │ │                 │ │              ▼
│ └─────────────────┘ │                  │ └─────────────────┘ │    ┌─────────────────────┐
└─────────────────────┘                  └─────────────────────┘    │ Playwright Browser  │
                                                   │                  │  (Microsoft Edge)   │
                                                   ▼                  │                     │
                                         ┌─────────────────────┐    │ ┌─────────────────┐ │
                                         │ Website Explorer    │    │ │ Real Browser    │ │
                                         │     Agent           │    │ │   Automation    │ │
                                         └─────────────────────┘    │ └─────────────────┘ │
                                                                     └─────────────────────┘
```

## 🎯 How It Works

### Phase 1: Discovery + Flow Selection
1. **Page Analysis**: Navigate to target URLs and analyze page structure
2. **Element Discovery**: Identify all interactive elements (buttons, forms, links)
3. **Hierarchical Flow Presentation**: Present discovered flows with clear numbering
4. **User Selection**: Support flexible selection formats (categories, ranges, keywords)

### Phase 2: Flow Validation
1. **Step-by-Step Execution**: Use browser automation to validate each flow
2. **Real-time Verification**: Visual confirmation of expected elements
3. **Dynamic Input Collection**: Request user data for forms and authentication
4. **Error Documentation**: Record issues and unexpected behaviors

### Phase 3: Iterative Expansion
1. **New Element Discovery**: Find additional functionality after flow execution
2. **Flow Expansion**: Propose additional flows based on discoveries
3. **Session Management**: Use tab management for parallel exploration

### Phase 4: Test Suite Documentation
1. **Comprehensive Documentation**: Generate detailed markdown test plans
2. **User-Selected Focus**: Document only validated user-chosen flows
3. **Strategic Recommendations**: Provide testing guidance and priorities

## 💻 Usage Examples

### Basic Navigation
```
"Navigate to https://playwright.microsoft.com"
```

### Element Discovery
```
"Take a snapshot and find all interactive elements"
```

### Flow Exploration
```
"Explore the authentication flows and document the login process"
```

### Advanced Exploration
```
"Navigate to the docs section, take screenshots of each page, and create a comprehensive test plan"
```

## 🛠️ Development

### Backend (FastAPI + AutoGen)
- **Location**: `backend/main.py`
- **Port**: 8000
- **Features**: WebSocket API, REST endpoints, AutoGen integration

### Frontend (React + TypeScript)
- **Location**: `frontend/src/App.tsx`  
- **Port**: 3000
- **Features**: Material-UI, WebSocket client, real-time updates

### Website Explorer Agent
- **Location**: `website_explore_agent.py`
- **Features**: Playwright automation, AutoGen tools, screenshot capture

## 🔧 Configuration

### Required Environment Variables
```bash
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"
AZURE_OPENAI_API_VERSION="2024-06-01"
AZURE_OPENAI_API_KEY="your-api-key"  # or use Azure AD
```

### Optional Frontend Environment Variables
```bash
REACT_APP_WS_URL="ws://localhost:8001/ws"
REACT_APP_API_URL="http://localhost:8001/api"
```

## 🚨 Troubleshooting

### Backend Issues
- **Agent not responding**: Check Azure OpenAI credentials and quota
- **Browser not opening**: Ensure Microsoft Edge is installed (`playwright install msedge`)
- **Connection errors**: Verify network connectivity and Azure endpoints

### Frontend Issues
- **WebSocket connection failed**: Check if backend is running on port 8000
- **Blank screenshots**: Ensure browser automation is working correctly
- **Build errors**: Run `npm install` in the frontend directory

### General Issues
- **Port conflicts**: Stop other services using ports 3000 or 8000
- **Missing dependencies**: Run installation commands for both frontend and backend
- **Environment variables**: Ensure all required variables are set correctly

## 📁 Project Structure

```
testing-autogen/
├── backend/
│   ├── main.py              # FastAPI server with WebSocket
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # Main React component
│   │   ├── App.css         # Styling
│   │   └── index.tsx       # React entry point
│   ├── public/
│   │   └── index.html      # HTML template
│   ├── package.json        # Node.js dependencies
│   └── README.md           # Frontend documentation
├── website_explore_agent.py # Core exploration agent
├── start.sh                # Full-stack startup script
└── README.md               # This file
```

## 🎯 Key Improvements Over Chainlit

1. **True Split-Screen Layout**: Dedicated panels instead of embedded content
2. **Better WebSocket Integration**: Real-time bidirectional communication
3. **Modern React UI**: Material-UI components with responsive design  
4. **Flexible Architecture**: Easy to extend and customize
5. **Better Performance**: Optimized for real-time browser automation
6. **Production Ready**: Proper error handling and reconnection logic

The system provides a seamless experience where you can interact with a powerful AutoGen agent through an intuitive web interface, making website exploration and test generation accessible and interactive with real-time visual feedback.

---

🚀 **Ready to explore?** Run `./start.sh` and open `http://localhost:3000`!
   ```bash
   export AZURE_OPENAI_ENDPOINT="your-azure-openai-endpoint"
   export AZURE_OPENAI_API_KEY="your-api-key"
   export AZURE_OPENAI_API_VERSION="2024-02-01"
   export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"
   ```

3. **Launch the UI**:
   ```bash
   chainlit run chainlit_autogen_ui.py
   ```

4. **Open Browser**: Navigate to `http://localhost:8001` to start chatting with your AutoGen agent

## How It Works

1. **Upload Files**: Drag and drop your recorded browser scripts or test files
2. **Start Conversation**: Ask the agent to generate tests or explain existing code
3. **Interactive Process**: The agent works in the background while you can:
   - Ask clarifying questions
   - Request modifications
   - Upload additional files
   - Get progress updates
4. **Download Results**: Generated test files are saved and can be downloaded

## Agent Capabilities

- Convert browser recordings to Playwright tests
- Generate comprehensive test suites with proper assertions
- Add error handling and retry logic
- Create modular, maintainable test code
- Handle dynamic content and wait conditions
- Generate test data and fixtures

## Configuration

The agent uses AutoGen's configuration system:
- **Model**: Azure OpenAI GPT-4
- **Temperature**: 0.1 for consistent code generation
- **Max Tokens**: 4000 for comprehensive responses
- **Tools**: File operations, test generation, project setup

## Example Usage

1. Upload a recorded browser script (JSON/HAR format)
2. Type: "Convert this recording to a Playwright test"
3. The agent will:
   - Analyze the recording
   - Generate a comprehensive test file
   - Add proper assertions and error handling
   - Create supporting files if needed
4. Ask follow-up questions like:
   - "Add data-driven test cases"
   - "Include mobile viewport testing"
   - "Generate test fixtures"

## Troubleshooting

- **Agent not responding**: Check Azure OpenAI credentials and quota
- **File upload issues**: Ensure files are in supported formats (JSON, HAR, Python)
- **Connection errors**: Verify network connectivity and Azure endpoints
- **Installation issues**: Use Python 3.9+ and update pip to latest version

## Development

To extend the agent:
1. Modify `autogen_latest_agent.py` for agent logic
2. Update `chainlit_autogen_ui.py` for UI changes
3. Add new tools in the agent's tool list
4. Test with `chainlit run chainlit_autogen_ui.py --debug`

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chainlit UI   │    │  AutoGen Agent  │    │  Azure OpenAI   │
│                 │◄──►│                 │◄──►│                 │
│ - Chat Interface│    │ - Test Gen      │    │ - GPT-4 Model   │
│ - File Upload   │    │ - Tool Calling  │    │ - Code Gen      │
│ - Session Mgmt  │    │ - Multi-Agent   │    │ - Analysis      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

The system provides a seamless experience where you can interact with a powerful AutoGen agent through an intuitive web interface, making test generation accessible and interactive.
