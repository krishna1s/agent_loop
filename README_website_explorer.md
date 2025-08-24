# Website Exploration Agent

An intelligent agent that systematically explores websites and generates comprehensive test plans using Microsoft AutoGen and Playwright MCP tools.

## 🎯 Overview

The Website Exploration Agent implements a sophisticated 4-phase iterative workflow to discover, validate, and document all possible user interactions on websites:

1. **Discovery + Selection**: Analyze pages, identify interactive elements, present flows hierarchically
2. **Flow Validation**: Execute selected flows step-by-step using Playwright MCP tools
3. **Iterative Expansion**: Discover new functionality and continue exploration
4. **Test Documentation**: Generate comprehensive test suite documentation

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Azure OpenAI account with GPT-4 deployment
- Playwright MCP tools (optional for full functionality)

### Installation

1. **Clone and Setup Environment**:
   ```bash
   cd /path/to/testing-autogen
   python -m venv env
   source env/bin/activate
   pip install chainlit autogen-agentchat 'autogen-ext[openai,azure]' azure-identity
   ```

2. **Configure Azure OpenAI**:
   ```bash
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
   export AZURE_OPENAI_API_KEY="your-api-key"
   export AZURE_OPENAI_API_VERSION="2024-02-01"
   export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"
   ```

3. **Launch the Agent**:
   ```bash
   ./run_website_explorer.sh
   ```
   
   Or manually:
   ```bash
   chainlit run website_explore_agent.py --port 8001
   ```

4. **Access the UI**: Open `http://localhost:8001` in your browser

## 🎯 Visual Browser Exploration

The Website Exploration Agent runs in **HEADED MODE** when exploring websites, meaning you'll see a real Chrome browser window open and perform the website exploration visually!

### 🌐 Smart Browser Management

- **On-Demand Browser Launch**: Browser only opens when you request website exploration (not on startup)
- **Visual Exploration**: Watch the agent navigate, click, and interact with websites in real-time
- **Single Browser Instance**: Reuses the same browser window for multiple explorations
- **Automatic Cleanup**: Browser closes when the session ends

### Browser Configuration

The agent launches with these settings for optimal visibility:
- **Headed Mode**: `headless: false` - Browser window is visible
- **DevTools**: Enabled for debugging and inspection
- **Slow Motion**: 500ms delays between actions for visibility
- **Maximized Window**: Starts in maximized state
- **On-Demand Launch**: Browser starts only when first MCP tool is called

### Browser Lifecycle

1. **🚀 Agent Startup**: Only the web UI starts, no browser yet
2. **💬 User Request**: You request website exploration
3. **🌐 Browser Launch**: Chrome browser opens in headed mode (first exploration only)
4. **🔍 Exploration**: Agent uses the browser to navigate and explore
5. **♻️ Reuse**: Subsequent explorations reuse the same browser
6. **🛑 Session End**: Browser automatically closes when you end the session

### Key Features in Action

1. **Lazy Browser Initialization**: Browser launches only when first MCP tool is called
2. **Real-time Navigation**: Watch the browser navigate to URLs you specify
3. **Element Discovery**: See elements get highlighted as they're discovered
4. **Interactive Actions**: Observe clicks, typing, and form interactions
5. **Visual Validation**: Screenshots automatically captured for evidence
6. **Error Visualization**: See exactly what goes wrong when issues occur

## 🔧 Features

### Core Capabilities
- **🕷️ Systematic Website Exploration**: Navigate and analyze website functionality
- **📋 Hierarchical Flow Discovery**: Organize user flows in clear hierarchical structure
- **✅ Interactive Flow Validation**: Execute flows step-by-step with real-time verification
- **📊 Comprehensive Test Plans**: Generate detailed test suite documentation
- **🔄 Iterative Discovery**: Continuously discover new functionality across sessions

### Playwright MCP Integration
- **Browser Automation**: Navigate, click, type, hover, select options
- **Element Discovery**: Capture accessibility snapshots for element identification
- **Session Management**: Multi-tab exploration with session preservation
- **Visual Verification**: Screenshots and visual evidence capture
- **Error Monitoring**: Console messages and network request monitoring

### User-Centric Design
- **Flexible Flow Selection**: Support multiple selection formats (categories, ranges, keywords)
- **Dynamic Input Collection**: Just-in-time requests for credentials and test data
- **Risk Assessment**: Clear risk indicators for destructive or expensive operations
- **Progress Tracking**: Real-time status updates and completion tracking

## 🎯 Usage Examples

### Basic Website Exploration
```
User: "Explore https://playwright.microsoft.com"

Agent Response:
🔍 Discovered Flows:

1. Authentication Flows
   1a. Sign in with Microsoft account [Medium Risk]
   1b. Sign in with work/school account [Medium Risk] 
   1c. Use another account option [Low Risk]

2. Workspace Management Flows
   2a. Access workspace settings [Low Risk]
   2b. Switch between workspaces [Medium Risk]
   2c. Create new workspace [High Risk - Creates resources]

3. Integration Setup Flows
   3a. Install Playwright Testing package [Low Risk]
   3b. Configure authentication [Medium Risk]
   3c. Set up Azure CLI [Low Risk]

Select flows to validate (e.g., "1a,2a" or "all"):
```

### Focused Exploration
```
User: "Explore the authentication flows on https://example.com and focus on login scenarios"

Agent: Will prioritize authentication-related functionality and provide detailed validation of login workflows.
```

### Custom Test Plan Generation
```
User: "Generate a test plan for the e-commerce checkout process"

Agent: Will discover, validate, and document the complete checkout workflow with payment validation steps.
```

## 📁 Project Structure

```
testing-autogen/
├── website_explore_agent.py      # Main agent implementation
├── run_website_explorer.sh       # Launch script
├── .chainlit/
│   └── config_website_explorer.toml  # UI configuration
├── test_plans/                   # Generated test documentation
│   ├── website_test_plan.md
│   └── flow_validation_results.json
└── README_website_explorer.md    # This file
```

## 🔄 Workflow Details

### Phase 1: Discovery + Selection
1. **Page Analysis**: Navigate to target URL using `navigate_to_url`
2. **Element Discovery**: Use `take_snapshot` to identify interactive elements
3. **Flow Generation**: Analyze elements and generate potential user flows
4. **Hierarchical Presentation**: Present flows with clear numbering system
5. **User Selection**: Support flexible selection formats

### Phase 2: Flow Validation
1. **Flow Breakdown**: Break selected flows into detailed steps
2. **Step Execution**: Use MCP tools (`click_element`, `type_text`, etc.)
3. **Real-time Verification**: Take snapshots to verify expected states
4. **Dynamic Input**: Request user data when needed (credentials, payment info)
5. **Error Handling**: Document issues and adjust flows accordingly

### Phase 3: Iterative Expansion
1. **New Discovery**: Identify new elements/pages after flow execution
2. **Session Management**: Use tab management for parallel exploration
3. **Flow Expansion**: Propose additional flows based on discoveries
4. **Loop Control**: Continue until complete coverage achieved

### Phase 4: Test Documentation
1. **User-Focused Documentation**: Document only validated user-selected flows
2. **Comprehensive Test Cases**: Step-by-step instructions with prerequisites
3. **Risk Assessment**: Priority matrix and gap analysis
4. **Automation Readiness**: Assess flows for automation potential

## 🛠️ Configuration

### Environment Variables
```bash
# Required
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"
AZURE_OPENAI_API_VERSION="2024-02-01"
AZURE_OPENAI_API_KEY="your-api-key"

# Optional (for Azure AD authentication)
AZURE_CLIENT_ID="your-client-id"
AZURE_CLIENT_SECRET="your-client-secret"
AZURE_TENANT_ID="your-tenant-id"
```

### Chainlit Configuration
The agent uses a custom Chainlit configuration (`config_website_explorer.toml`) with:
- Website exploration-specific UI settings
- Extended session timeout for long explorations
- Multi-modal support for file uploads
- Custom theming for exploration workflows

## 📊 Output Examples

### Generated Test Plan Structure
```markdown
# Test Suite: Example Website
## Overview
- Application functionality summary
- Test coverage scope (user-selected flows only)
- Testing recommendations

## Test Categories
### 1. Authentication Tests
- Test Case 1.1: User Login
  - **Objective**: Verify user can successfully log in
  - **Prerequisites**: Valid user credentials
  - **Test Steps**: [Detailed step-by-step instructions]
  - **Expected Results**: User authenticated and redirected
  - **Priority**: Critical
  - **Automation**: Ready

### 2. Workspace Management Tests
- Test Case 2.1: Create New Workspace
  - **Prerequisites**: Complete Test Case 1.1 (or ensure logged in)
  - **Test Steps**: [Complete workflow from authenticated state]
  - **Dependencies**: Requires authentication flow
  - **Priority**: High

## Test Dependencies Map
- **Independent Tests**: 1.1
- **Dependent Tests**: 2.1 → requires 1.1
- **Test Chains**: 1.1 → 2.1 → 2.2

## Recommendations
- Automation strategy based on user selections
- Risk assessment for selected flows
- Maintenance guidelines
```

## 🔍 Flow Selection Formats

The agent supports multiple selection formats for maximum flexibility:

- **Category Level**: `"1"` (all sub-flows in category 1)
- **Specific Sub-flows**: `"1a,1c"` (login and password reset only)
- **Mixed Selection**: `"1a,2,3b"` (specific + entire category + specific)
- **Range Selection**: `"1a-1c"` (all sub-flows from 1a to 1c)
- **Keyword Selection**: `"login,workspace"` (semantic matching)
- **Complete Coverage**: `"all"` (everything discovered)

## 🚨 Risk Assessment

The agent categorizes flows by risk level:
- **🟢 Low Risk**: Read-only operations, navigation, information display
- **🟡 Medium Risk**: Account modifications, state changes, data entry
- **🔴 High Risk**: Destructive operations, purchases, account deletions

## 🤝 Contributing

To extend the Website Exploration Agent:

1. **Add New MCP Tools**: Implement additional Playwright MCP functions
2. **Enhance Flow Logic**: Improve flow discovery and validation algorithms
3. **Custom Documentation**: Add new documentation formats or templates
4. **UI Improvements**: Enhance the Chainlit interface for better UX

## 📝 License

This project follows the same license as the parent testing-autogen repository.

## 🆘 Troubleshooting

### Common Issues

1. **Agent not responding**:
   - Check Azure OpenAI credentials and quota
   - Verify network connectivity to Azure endpoints

2. **MCP tools not working**:
   - Ensure Playwright MCP server is running
   - Check MCP tool configurations

3. **Flow validation failures**:
   - Review console messages for JavaScript errors
   - Check network requests for failed API calls
   - Verify element selectors are still valid

4. **Memory/Performance issues**:
   - Close unused browser tabs using `close_tab`
   - Clear session data between explorations
   - Reduce concurrent flow validations

### Debug Mode

Enable debug logging by setting:
```bash
export CHAINLIT_DEBUG=true
```

This provides detailed logs of:
- MCP tool execution
- Flow validation steps
- Agent decision making
- Session management

## 🎉 Success Stories

The Website Exploration Agent has been successfully used to:
- Document complex e-commerce checkout flows
- Generate comprehensive authentication test suites
- Discover hidden functionality in SaaS applications
- Create automation-ready test cases for CI/CD pipelines
- Validate accessibility and responsive design flows
