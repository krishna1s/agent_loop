import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  CircularProgress,
  Divider,
  Chip,
  Tabs,
  Tab,
  Drawer,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  IconButton,
  Menu,
  MenuItem
} from '@mui/material';
import {
  Send as SendIcon,
  Screenshot as ScreenshotIcon,
  NavigateNext as NavigateIcon,
  Code as CodeIcon,
  Menu as MenuIcon,
  Add as AddIcon,
  MoreVert as MoreVertIcon,
  Delete as DeleteIcon,
  Chat as ChatIcon
} from '@mui/icons-material';
import './App.css';

interface Message {
  id: string;
  type: 'user' | 'agent' | 'system' | 'error';
  content: string;
  timestamp: Date;
  screenshot?: string;
}

interface ApiResponse {
  success: boolean;
  response?: string;
  thread_id?: string;
  error?: string;
}

interface Thread {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
}

const SIDEBAR_WIDTH = 280;

function App() {
  // API base URL (move to top to avoid use-before-define)
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';
  // Polling interval in milliseconds
  const MESSAGE_POLL_INTERVAL = 2000;
  // State management
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentScreenshot, setCurrentScreenshot] = useState<string | null>(null);
  const [agentStatus, setAgentStatus] = useState<string>('Initializing...');
  const [tabValue, setTabValue] = useState(0);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [availableThreads, setAvailableThreads] = useState<Thread[]>([]);
  const [isLoadingThread, setIsLoadingThread] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedThreadForMenu, setSelectedThreadForMenu] = useState<string | null>(null);
  
  // WebSocket for screenshots
  const screenshotWebSocketRef = useRef<WebSocket | null>(null);
  
  // References
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Helper to compare messages arrays (by id and content)
  const areMessagesEqual = (msgs1: Message[], msgs2: Message[]) => {
    if (msgs1.length !== msgs2.length) return false;
    for (let i = 0; i < msgs1.length; i++) {
      if (msgs1[i].id !== msgs2[i].id || msgs1[i].content !== msgs2[i].content) {
        return false;
      }
    }
    return true;
  };

  // Poll for new messages in the current thread
  useEffect(() => {
    if (!currentThreadId) return;
    let isMounted = true;
    const pollMessages = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/threads/${currentThreadId}/messages`);
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.messages) {
            const formattedMessages = data.messages.map((msg: any) => ({
              id: msg.id || Date.now().toString() + Math.random().toString(36).substr(2, 9),
              type: msg.type || (msg.role === 'user' ? 'user' : 'agent'),
              content: msg.content || msg.message,
              timestamp: new Date(msg.timestamp || Date.now())
            }));
            if (isMounted && !areMessagesEqual(formattedMessages, messages)) {
              setMessages(formattedMessages);
            }
          }
        }
      } catch (error) {
        // Ignore polling errors
      }
    };
    const interval = setInterval(pollMessages, MESSAGE_POLL_INTERVAL);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentThreadId, API_BASE_URL]);

  // Initialize screenshot WebSocket connection
  const initializeScreenshotWebSocket = useCallback(() => {
    // Always use ws://localhost:8001/ws/screenshots for backend compatibility
    const wsUrl = 'ws://localhost:8001/ws/screenshots';
    console.log('🔌 Connecting to screenshot WebSocket:', wsUrl);
    screenshotWebSocketRef.current = new WebSocket(wsUrl);
    
    screenshotWebSocketRef.current.onopen = () => {
      console.log('📸 Screenshot WebSocket connected');
    };
    
    screenshotWebSocketRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'screenshot' && data.data) {
          console.log('📸 Received screenshot update');
          setCurrentScreenshot(data.data);
          // Auto-switch to Live View tab when screenshot is received
          setTabValue(0);
        }
      } catch (error) {
        console.error('Error parsing screenshot WebSocket message:', error);
      }
    };
    
    screenshotWebSocketRef.current.onclose = () => {
      console.log('📸 Screenshot WebSocket disconnected');
      // Attempt to reconnect after 3 seconds
      setTimeout(() => {
        if (screenshotWebSocketRef.current?.readyState === WebSocket.CLOSED) {
          initializeScreenshotWebSocket();
        }
      }, 3000);
    };
    
    screenshotWebSocketRef.current.onerror = (error) => {
      console.error('Screenshot WebSocket error:', error);
    };
  }, [API_BASE_URL]);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Initialize screenshot WebSocket on component mount
  useEffect(() => {
    initializeScreenshotWebSocket();
    
    return () => {
      if (screenshotWebSocketRef.current) {
        screenshotWebSocketRef.current.close();
      }
    };
  }, [initializeScreenshotWebSocket]);

  // Load all available threads
  const loadAvailableThreads = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/threads`);
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setAvailableThreads(data.threads);
          console.log(`📋 Loaded ${data.threads.length} available threads`);
        }
      }
    } catch (error) {
      console.error('Failed to load available threads:', error);
    }
  }, [API_BASE_URL]);

  // Create a new thread
  const createThread = useCallback(async (): Promise<string | null> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/threads`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: `Exploration_${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}`
        }),
      });

      if (response.ok) {
        const data = await response.json();
        return data.thread_id;
      }
    } catch (error) {
      console.error('Failed to create thread:', error);
    }
    return null;
  }, [API_BASE_URL]);

  const addMessage = (message: Omit<Message, 'id'>) => {
    setMessages(prev => [...prev, {
      ...message,
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9)
    }]);
  };

  // Initialize or load existing thread
  const initializeThread = useCallback(async () => {
    setIsLoadingThread(true);
    try {
      // Try to get existing threads first
      const threadsResponse = await fetch(`${API_BASE_URL}/api/threads`);
      
      if (threadsResponse.ok) {
        const threadsData = await threadsResponse.json();
        if (threadsData.success && threadsData.threads.length > 0) {
          // Use the most recent thread
          const mostRecentThread = threadsData.threads[0];
          setCurrentThreadId(mostRecentThread.id);
          setAvailableThreads(threadsData.threads);
          
          // Load messages from this thread
          try {
            const response = await fetch(`${API_BASE_URL}/api/threads/${mostRecentThread.id}/messages`);
            if (response.ok) {
              const data = await response.json();
              if (data.success && data.messages) {
                const formattedMessages = data.messages.map((msg: any) => ({
                  id: msg.id || Date.now().toString() + Math.random().toString(36).substr(2, 9),
                  type: msg.type || (msg.role === 'user' ? 'user' : 'agent'),
                  content: msg.content || msg.message,
                  timestamp: new Date(msg.timestamp || Date.now())
                }));
                setMessages(formattedMessages);
                console.log(`📨 Loaded ${formattedMessages.length} messages from thread ${mostRecentThread.id}`);
              }
            }
          } catch (error) {
            console.error('Failed to load thread messages:', error);
          }
          
          console.log(`🧵 Loaded existing thread: ${mostRecentThread.id}`);
        } else {
          // Create a new thread if none exist - inline implementation
          const threadId = await createThread();
          if (threadId) {
            setCurrentThreadId(threadId);
            setMessages([]);
            
            addMessage({
              type: 'system',
              content: '🧠 Welcome to a new exploration session! I can help you navigate and explore websites, then generate comprehensive test documentation. Try asking me to explore a URL.',
              timestamp: new Date()
            });
            
            await loadAvailableThreads();
            console.log(`🧵 Created new thread: ${threadId}`);
          }
        }
      } else {
        // Create a new thread if we can't fetch existing ones - inline implementation
        const threadId = await createThread();
        if (threadId) {
          setCurrentThreadId(threadId);
          setMessages([]);
          
          addMessage({
            type: 'system',
            content: '🧠 Welcome to a new exploration session! I can help you navigate and explore websites, then generate comprehensive test documentation. Try asking me to explore a URL.',
            timestamp: new Date()
          });
          
          await loadAvailableThreads();
          console.log(`🧵 Created new thread: ${threadId}`);
        }
      }
    } catch (error) {
      console.error('Failed to initialize thread:', error);
      // Fallback: create a new thread - inline implementation
      const threadId = await createThread();
      if (threadId) {
        setCurrentThreadId(threadId);
        setMessages([]);
        
        addMessage({
          type: 'system',
          content: '🧠 Welcome to a new exploration session! I can help you navigate and explore websites, then generate comprehensive test documentation. Try asking me to explore a URL.',
          timestamp: new Date()
        });
        
        await loadAvailableThreads();
        console.log(`🧵 Created new thread: ${threadId}`);
      }
    } finally {
      setIsLoadingThread(false);
    }
  }, [API_BASE_URL, createThread, loadAvailableThreads]);

  // Health check and initialization
  useEffect(() => {
    // Health check function
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
          const data = await response.json();
          setIsConnected(prev => {
            const wasConnected = prev;
            if (data.agent_ready && !wasConnected) {
              addMessage({
                type: 'system',
                content: '🌐 Connected to Website Exploration Agent! Ready to explore websites and generate test plans.',
                timestamp: new Date()
              });
            }
            return data.agent_ready;
          });
          setAgentStatus(data.agent_ready ? 'Agent ready' : 'Agent initializing...');
        } else {
          setIsConnected(prev => {
            if (prev) {
              addMessage({
                type: 'error',
                content: 'Lost connection to the agent. Attempting to reconnect...',
                timestamp: new Date()
              });
            }
            return false;
          });
          setAgentStatus('Connection failed');
        }
      } catch (error) {
        console.error('Health check failed:', error);
        setIsConnected(prev => {
          if (prev) {
            addMessage({
              type: 'error',
              content: 'Lost connection to the agent. Attempting to reconnect...',
              timestamp: new Date()
            });
          }
          return false;
        });
        setAgentStatus('Connection error');
      }
    };

    // Initial health check and thread initialization
    const initializeApp = async () => {
      await checkHealth();
      await initializeThread();
    };

    initializeApp();

    // Set up periodic health check every 30 seconds
    const healthCheckInterval = setInterval(checkHealth, 30000);

    return () => {
      clearInterval(healthCheckInterval);
    };
  }, [API_BASE_URL, initializeThread]);

  // Create a new thread with welcome message
  const createNewThread = useCallback(async () => {
    const threadId = await createThread();
    if (threadId) {
      setCurrentThreadId(threadId);
      setMessages([]); // Clear current messages
      
      // Add welcome message for new thread
      addMessage({
        type: 'system',
        content: '🧠 Welcome to a new exploration session! I can help you navigate and explore websites, then generate comprehensive test documentation. Try asking me to explore a URL.',
        timestamp: new Date()
      });
      
      // Refresh thread list
      await loadAvailableThreads();
      
      console.log(`🧵 Created new thread: ${threadId}`);
    }
  }, [createThread, loadAvailableThreads]);

  // Load messages from a specific thread
  const loadThreadMessages = useCallback(async (threadId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/threads/${threadId}/messages`);
      if (response.ok) {
        const data = await response.json();
        if (data.success && data.messages) {
          // Convert backend messages to frontend format
          const formattedMessages = data.messages.map((msg: any) => ({
            id: msg.id || Date.now().toString() + Math.random().toString(36).substr(2, 9),
            type: msg.type || (msg.role === 'user' ? 'user' : 'agent'),
            content: msg.content || msg.message,
            timestamp: new Date(msg.timestamp || Date.now())
          }));
          setMessages(formattedMessages);
          console.log(`📨 Loaded ${formattedMessages.length} messages from thread ${threadId}`);
        }
      }
    } catch (error) {
      console.error('Failed to load thread messages:', error);
    }
  }, [API_BASE_URL]);

  // Switch to a different thread
  const switchToThread = async (threadId: string) => {
    try {
      setIsLoadingThread(true);
      setCurrentThreadId(threadId);
      await loadThreadMessages(threadId);
      console.log(`🔄 Switched to thread: ${threadId}`);
    } catch (error) {
      console.error('Failed to switch thread:', error);
    } finally {
      setIsLoadingThread(false);
    }
  };

  // Delete a thread
  const deleteThread = async (threadId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/threads/${threadId}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        // Refresh thread list
        await loadAvailableThreads();
        
        // If we deleted the current thread, create a new one
        if (threadId === currentThreadId) {
          await createNewThread();
        }
        
        console.log(`🗑️ Deleted thread: ${threadId}`);
      }
    } catch (error) {
      console.error('Failed to delete thread:', error);
    }
  };

  // Send message via streaming API
  const sendMessageViaStreamingAPI = async (message: string, threadId: string | null = null): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/message/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          thread_id: threadId
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body reader available');
      }

      let streamingMessage = '';
      
      // Add initial streaming message
      const streamingMessageObject: Message = {
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
        type: 'agent',
        content: '🤖 Agent is working...',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, streamingMessageObject]);
      const streamingMessageId = streamingMessageObject.id;

      try {
        while (true) {
          const { done, value } = await reader.read();
          
          if (done) break;
          
          const chunk = new TextDecoder().decode(value);
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                if (data.error) {
                  addMessage({
                    type: 'error',
                    content: data.error,
                    timestamp: new Date()
                  });
                  return;
                }
                
                if (data.content) {
                  streamingMessage += data.content;
                  
                  // Update the streaming message in real-time
                  setMessages(prev => {
                    const updated = [...prev];
                    const messageIndex = updated.findIndex(msg => msg.id === streamingMessageId);
                    
                    if (messageIndex >= 0) {
                      updated[messageIndex] = {
                        ...updated[messageIndex],
                        content: streamingMessage,
                        timestamp: new Date()
                      };
                    }
                    return updated;
                  });
                }
                
                if (data.complete) {
                  console.log('🎯 Streaming completed');
                  setAgentStatus(`✅ Response completed (Thread: ${threadId?.substring(0, 8)}...)`);
                  return;
                }
                
              } catch (parseError) {
                console.error('Error parsing streaming data:', parseError);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }

    } catch (error) {
      console.error('Streaming error:', error);
      addMessage({
        type: 'error',
        content: `Streaming error: ${error}`,
        timestamp: new Date()
      });
    }
  };

  // Send message via API (fallback to regular API)
  const sendMessageViaAPI = async (message: string, threadId: string | null = null): Promise<ApiResponse> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          thread_id: threadId
        }),
      });

      if (response.ok) {
        const data = await response.json();
        return data;
      } else {
        return { success: false, error: `HTTP ${response.status}: ${response.statusText}` };
      }
    } catch (error) {
      return { success: false, error: `Network error: ${error}` };
    }
  };

  const sendMessage = async () => {
    if (!currentMessage.trim() || !isConnected || isLoading) return;

    // Add user message to chat
    addMessage({
      type: 'user',
      content: currentMessage,
      timestamp: new Date()
    });

    setIsLoading(true);
    setAgentStatus('Processing your request...');

    try {
      // Ensure we have a thread for context persistence
      let threadId = currentThreadId;
      if (!threadId) {
        console.log('🧵 No thread ID found, creating new thread for context persistence');
        threadId = await createThread();
        setCurrentThreadId(threadId);
        console.log(`🧵 Created thread for context: ${threadId}`);
      } else {
        console.log(`🧵 Using existing thread for context: ${threadId}`);
      }

      // Use streaming API for real-time feedback
      await sendMessageViaStreamingAPI(currentMessage, threadId);

    } catch (error) {
      addMessage({
        type: 'error',
        content: `Error: ${error}`,
        timestamp: new Date()
      });
      setAgentStatus('Error occurred');
    } finally {
      setIsLoading(false);
    }

    setCurrentMessage('');
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  const quickActions = [
    { label: 'Navigate to URL', action: 'navigate', placeholder: 'https://example.com' },
    { label: 'Take Snapshot', action: 'snapshot', placeholder: '' },
    { label: 'Get Page Info', action: 'info', placeholder: '' }
  ];

  const sendQuickAction = async (action: string, placeholder: string) => {
    const message = placeholder ? 
      `${action === 'navigate' ? 'Navigate to' : action} ${placeholder}` :
      `Take a snapshot of the current page`;
    
    addMessage({
      type: 'user',
      content: message,
      timestamp: new Date()
    });

    setIsLoading(true);
    setAgentStatus('Processing quick action...');

    try {
      // Get or create thread ID
      let threadId = currentThreadId;
      if (!threadId) {
        threadId = await createThread();
        setCurrentThreadId(threadId);
      }

      // Send message via API
      const result = await sendMessageViaAPI(message, threadId);

      if (result.success && result.response) {
        addMessage({
          type: 'agent',
          content: result.response,
          timestamp: new Date()
        });
        setAgentStatus('Quick action completed');
      } else {
        addMessage({
          type: 'error',
          content: result.error || 'Failed to execute quick action',
          timestamp: new Date()
        });
        setAgentStatus('Quick action failed');
      }
    } catch (error) {
      addMessage({
        type: 'error',
        content: `Error: ${error}`,
        timestamp: new Date()
      });
      setAgentStatus('Quick action failed');
    } finally {
      setIsLoading(false);
    }
  };

  // Menu handlers
  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, threadId: string) => {
    setMenuAnchorEl(event.currentTarget);
    setSelectedThreadForMenu(threadId);
  };

  const handleMenuClose = () => {
    setMenuAnchorEl(null);
    setSelectedThreadForMenu(null);
  };

  const handleDeleteFromMenu = async () => {
    if (selectedThreadForMenu) {
      await deleteThread(selectedThreadForMenu);
    }
    handleMenuClose();
  };

  // Format thread name for display
  const formatThreadName = (thread: Thread) => {
    if (thread.name.startsWith('Exploration_')) {
      const dateStr = thread.name.replace('Exploration_', '').replace(/-/g, ':');
      try {
        return new Date(dateStr).toLocaleDateString() + ' ' + new Date(dateStr).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
      } catch {
        return thread.name;
      }
    }
    return thread.name;
  };

  return (
    <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* Left Sidebar - Thread Management */}
      <Drawer
        variant="persistent"
        anchor="left"
        open={sidebarOpen}
        sx={{
          width: sidebarOpen ? SIDEBAR_WIDTH : 0,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: SIDEBAR_WIDTH,
            boxSizing: 'border-box',
            borderRight: '1px solid #e0e0e0',
          },
        }}
      >
        <Box sx={{ p: 2, borderBottom: '1px solid #e0e0e0' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <ChatIcon />
              Conversations
            </Typography>
            <IconButton onClick={() => setSidebarOpen(false)} size="small">
              <MenuIcon />
            </IconButton>
          </Box>
          
          <Button
            fullWidth
            variant="outlined"
            startIcon={<AddIcon />}
            onClick={createNewThread}
            disabled={isLoading || isLoadingThread}
            sx={{ mb: 1 }}
          >
            New Chat
          </Button>
        </Box>

        <List sx={{ flex: 1, overflow: 'auto', py: 0 }}>
          {availableThreads.map((thread) => (
            <ListItem key={thread.id} disablePadding>
              <ListItemButton
                selected={thread.id === currentThreadId}
                onClick={() => switchToThread(thread.id)}
                sx={{
                  py: 2,
                  px: 2,
                  '&.Mui-selected': {
                    backgroundColor: 'primary.light',
                    color: 'primary.contrastText',
                    '&:hover': {
                      backgroundColor: 'primary.main',
                    }
                  }
                }}
              >
                <ListItemText
                  primary={formatThreadName(thread)}
                  secondary={new Date(thread.updated_at).toLocaleDateString()}
                  primaryTypographyProps={{
                    variant: 'body2',
                    sx: { 
                      fontWeight: thread.id === currentThreadId ? 'bold' : 'normal',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap'
                    }
                  }}
                  secondaryTypographyProps={{
                    variant: 'caption',
                    sx: { opacity: 0.7 }
                  }}
                />
                <IconButton
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleMenuOpen(e, thread.id);
                  }}
                  sx={{ ml: 1 }}
                >
                  <MoreVertIcon fontSize="small" />
                </IconButton>
              </ListItemButton>
            </ListItem>
          ))}
        </List>

        <Menu
          anchorEl={menuAnchorEl}
          open={Boolean(menuAnchorEl)}
          onClose={handleMenuClose}
        >
          <MenuItem onClick={handleDeleteFromMenu} sx={{ color: 'error.main' }}>
            <DeleteIcon sx={{ mr: 1 }} />
            Delete Chat
          </MenuItem>
        </Menu>
      </Drawer>

      {/* Main Content Area */}
      <Box sx={{ 
        flex: 1, 
        display: 'flex', 
        transition: 'margin-left 0.3s',
        marginLeft: sidebarOpen ? 0 : `-${SIDEBAR_WIDTH}px`,
        overflow: 'hidden'
      }}>
        {/* Toggle Sidebar Button (when closed) */}
        {!sidebarOpen && (
          <IconButton
            onClick={() => setSidebarOpen(true)}
            sx={{
              position: 'fixed',
              top: 16,
              left: 16,
              zIndex: 1000,
              backgroundColor: 'primary.main',
              color: 'white',
              '&:hover': {
                backgroundColor: 'primary.dark',
              }
            }}
          >
            <MenuIcon />
          </IconButton>
        )}

        {/* Chat Interface */}
        <Box sx={{ 
          width: '50%', 
          display: 'flex', 
          flexDirection: 'column', 
          bgcolor: '#f5f5f5',
          height: '100vh',
          overflow: 'hidden'
        }}>
          <Paper sx={{ p: 2, m: 1, bgcolor: 'primary.main', color: 'white', flexShrink: 0 }}>
            <Typography variant="h6" component="h1">
              🕷️ Website Exploration Agent
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 1, flexWrap: 'wrap', gap: 1 }}>
              <Chip 
                label={isConnected ? 'Connected' : 'Disconnected'} 
                color={isConnected ? 'success' : 'error'}
                size="small"
              />
              {currentThreadId && (
                <Chip 
                  label={`Thread: ${currentThreadId.substring(0, 8)}...`}
                  color="info"
                  size="small"
                />
              )}
              {isLoadingThread && (
                <Chip 
                  label="Loading thread..."
                  color="warning"
                  size="small"
                />
              )}
              <Typography variant="body2" sx={{ opacity: 0.9, width: '100%', mt: 1 }}>
                {agentStatus}
              </Typography>
            </Box>
          </Paper>

          {/* Quick Actions */}
          <Paper sx={{ m: 1, p: 2, flexShrink: 0 }}>
            <Typography variant="subtitle2" gutterBottom>Quick Actions:</Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {quickActions.map((action) => (
                <Button
                  key={action.action}
                  size="small"
                  variant="outlined"
                  onClick={() => sendQuickAction(action.action, action.placeholder)}
                  disabled={!isConnected || isLoading}
                >
                  {action.label}
                </Button>
              ))}
            </Box>
          </Paper>

          {/* Messages Area */}
          <Paper sx={{ 
            flex: 1, 
            m: 1, 
            display: 'flex', 
            flexDirection: 'column',
            overflow: 'hidden',
            minHeight: 0
          }}>
            <Box sx={{ 
              flex: 1, 
              overflowY: 'auto', 
              p: 2,
              minHeight: 0
            }}>
              {messages.map((message) => (
                <Box
                  key={message.id}
                  sx={{
                    mb: 2,
                    p: 2,
                    borderRadius: 2,
                    bgcolor: message.type === 'user' ? 'primary.light' : 
                            message.type === 'error' ? 'error.light' :
                            message.type === 'system' ? 'info.light' : 'grey.100',
                    color: message.type === 'user' ? 'white' : 'text.primary',
                    ml: message.type === 'user' ? 4 : 0,
                    mr: message.type === 'user' ? 0 : 4
                  }}
                >
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="caption" sx={{ 
                      color: message.type === 'user' ? 'rgba(255,255,255,0.8)' : 'text.secondary',
                      fontWeight: 'bold'
                    }}>
                      {message.type === 'user' ? 'You' : 
                       message.type === 'agent' ? '🤖 Agent' :
                       message.type === 'system' ? '⚙️ System' : '❌ Error'}
                    </Typography>
                    <Typography variant="caption" sx={{ 
                      color: message.type === 'user' ? 'rgba(255,255,255,0.6)' : 'text.secondary'
                    }}>
                      {message.timestamp.toLocaleTimeString()}
                    </Typography>
                  </Box>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                    {message.content}
                  </Typography>
                </Box>
              ))}
              <div ref={messagesEndRef} />
            </Box>

            {/* Input Area */}
            <Divider />
            <Box sx={{ p: 2, flexShrink: 0 }}>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  fullWidth
                  multiline
                  maxRows={4}
                  placeholder="Type your exploration request... (e.g., 'Navigate to https://example.com')"
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  disabled={!isConnected || isLoading}
                  variant="outlined"
                  size="small"
                />
                <Button
                  variant="contained"
                  onClick={sendMessage}
                  disabled={!currentMessage.trim() || !isConnected || isLoading}
                  sx={{ minWidth: 48 }}
                >
                  {isLoading ? <CircularProgress size={20} /> : <SendIcon />}
                </Button>
              </Box>
            </Box>
          </Paper>
        </Box>

        {/* Right Panel - Browser View */}
        <Box sx={{ 
          width: '50%', 
          display: 'flex', 
          flexDirection: 'column', 
          bgcolor: '#fff',
          height: '100vh',
          overflow: 'hidden'
        }}>
          <Paper sx={{ p: 2, m: 1, bgcolor: 'secondary.main', color: 'white', flexShrink: 0 }}>
            <Typography variant="h6" component="h2">
              🌐 Live Browser View
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.9 }}>
              Real-time website exploration and element discovery
            </Typography>
          </Paper>

          {/* Browser Screenshot Display */}
          <Paper sx={{ 
            flex: 1, 
            m: 1, 
            display: 'flex', 
            flexDirection: 'column',
            overflow: 'hidden',
            minHeight: 0
          }}>
            <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider', flexShrink: 0 }}>
              <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
                <Tab label="Live View" icon={<ScreenshotIcon />} />
                <Tab label="Analysis" icon={<CodeIcon />} />
              </Tabs>
            </Box>

            <Box sx={{ flex: 1, overflow: 'auto', position: 'relative', minHeight: 0 }}>
              {tabValue === 0 && (
                <Box sx={{ 
                  height: '100%', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  bgcolor: '#f8f9fa',
                  minHeight: '400px'
                }}>
                  {currentScreenshot ? (
                    <Box sx={{ 
                      width: '100%', 
                      height: '100%', 
                      overflow: 'auto',
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'flex-start',
                      p: 1
                    }}>
                      <img
                        src={`data:image/png;base64,${currentScreenshot}`}
                        alt="Browser Screenshot"
                        style={{
                          maxWidth: '100%',
                          height: 'auto',
                          border: '1px solid #ddd',
                          borderRadius: '4px',
                          boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                        }}
                      />
                    </Box>
                  ) : (
                    <Box sx={{ textAlign: 'center', color: 'text.secondary' }}>
                      <ScreenshotIcon sx={{ fontSize: 64, mb: 2, opacity: 0.3 }} />
                      <Typography variant="h6" gutterBottom>
                        No Browser View Available
                      </Typography>
                      <Typography variant="body2">
                        Start exploring a website to see live browser screenshots here
                      </Typography>
                      <Button
                        variant="outlined"
                        startIcon={<NavigateIcon />}
                        sx={{ mt: 2 }}
                        onClick={() => sendQuickAction('navigate', 'https://example.com')}
                        disabled={!isConnected}
                      >
                        Navigate to Example.com
                      </Button>
                    </Box>
                  )}
                </Box>
              )}

              {tabValue === 1 && (
                <Box sx={{ p: 2, height: '100%', overflow: 'auto' }}>
                  <Typography variant="h6" gutterBottom>
                    🔍 Element Analysis
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Element discovery and interaction details will appear here when available.
                  </Typography>
                </Box>
              )}
            </Box>
          </Paper>
        </Box>
      </Box>
    </Box>
  );
}

export default App;
