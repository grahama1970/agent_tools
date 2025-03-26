import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { User, AdminUser, ComponentProps, ApiResponse } from '../interfaces/models';

/**
 * Props for the UserProfile component
 */
interface UserProfileProps extends ComponentProps<User> {
  userId: string;
  showAdminControls?: boolean;
  onUserUpdate?: (user: User) => void;
}

/**
 * Component state interface
 */
interface UserProfileState {
  isEditing: boolean;
  user: User | null;
  isLoading: boolean;
  error: string | null;
}

/**
 * Higher-order component that adds authentication to a component
 * @param WrappedComponent Component to wrap with authentication
 * @returns Wrapped component with authentication
 */
export function withAuthentication<P extends object>(
  WrappedComponent: React.ComponentType<P>
): React.FC<P & { isAuthenticated?: boolean }> {
  return function WithAuthentication(props: P & { isAuthenticated?: boolean }) {
    const { isAuthenticated = false, ...rest } = props;
    
    if (!isAuthenticated) {
      return <div>Please log in to view this content</div>;
    }
    
    return <WrappedComponent {...(rest as P)} />;
  };
}

/**
 * Component for displaying and editing user profiles
 */
export class UserProfile extends React.Component<UserProfileProps, UserProfileState> {
  private updateTimer: NodeJS.Timeout | null = null;
  
  /**
   * Constructor initializes component state
   */
  constructor(props: UserProfileProps) {
    super(props);
    
    this.state = {
      isEditing: false,
      user: props.data || null,
      isLoading: props.isLoading || false,
      error: null
    };
    
    // Method bindings
    this.handleEdit = this.handleEdit.bind(this);
    this.handleSave = this.handleSave.bind(this);
    this.handleCancel = this.handleCancel.bind(this);
    this.handleFieldChange = this.handleFieldChange.bind(this);
  }
  
  /**
   * Lifecycle method that updates state when props change
   */
  componentDidUpdate(prevProps: UserProfileProps) {
    if (prevProps.data !== this.props.data) {
      this.setState({ user: this.props.data || null });
    }
    
    if (prevProps.isLoading !== this.props.isLoading) {
      this.setState({ isLoading: this.props.isLoading || false });
    }
  }
  
  /**
   * Clean up timer on component unmount
   */
  componentWillUnmount() {
    if (this.updateTimer) {
      clearTimeout(this.updateTimer);
    }
  }
  
  /**
   * Handle edit button click
   */
  handleEdit() {
    this.setState({ isEditing: true });
  }
  
  /**
   * Handle save button click
   */
  handleSave() {
    const { user } = this.state;
    const { onUserUpdate } = this.props;
    
    if (user) {
      this.setState({ isLoading: true });
      
      // Simulate API call
      this.updateTimer = setTimeout(() => {
        this.setState({ 
          isLoading: false,
          isEditing: false 
        });
        
        if (onUserUpdate) {
          onUserUpdate(user);
        }
      }, 1000);
    }
  }
  
  /**
   * Handle cancel button click
   */
  handleCancel() {
    this.setState({ 
      isEditing: false,
      user: this.props.data || null
    });
  }
  
  /**
   * Handle input field changes
   */
  handleFieldChange(event: React.ChangeEvent<HTMLInputElement>) {
    const { name, value } = event.target;
    
    this.setState(prevState => ({
      user: prevState.user ? {
        ...prevState.user,
        [name]: value
      } : null
    }));
  }
  
  /**
   * Render admin controls if user is admin
   */
  renderAdminControls() {
    const { user } = this.state;
    
    if (!user || !this.isAdmin(user)) {
      return null;
    }
    
    return (
      <div className="admin-controls">
        <h3>Admin Controls</h3>
        <div>
          <label>Permissions:</label>
          <ul>
            {(user as AdminUser).permissions.map(perm => (
              <li key={perm}>{perm}</li>
            ))}
          </ul>
        </div>
      </div>
    );
  }
  
  /**
   * Check if user is an admin
   */
  isAdmin(user: User): user is AdminUser {
    return user.roles.includes('admin') && 'permissions' in user;
  }
  
  /**
   * Main render method
   */
  render() {
    const { isEditing, user, isLoading, error } = this.state;
    const { showAdminControls, className, style } = this.props;
    
    if (isLoading) {
      return <div className="loading">Loading user data...</div>;
    }
    
    if (error) {
      return <div className="error">{error}</div>;
    }
    
    if (!user) {
      return <div className="no-data">No user data available</div>;
    }
    
    return (
      <div className={`user-profile ${className || ''}`} style={style}>
        <h2>User Profile</h2>
        
        {isEditing ? (
          <form onSubmit={e => { e.preventDefault(); this.handleSave(); }}>
            <div>
              <label>Username:</label>
              <input
                type="text"
                name="username"
                value={user.username}
                onChange={this.handleFieldChange}
                disabled
              />
            </div>
            <div>
              <label>Email:</label>
              <input
                type="email"
                name="email"
                value={user.email}
                onChange={this.handleFieldChange}
              />
            </div>
            <div>
              <label>First Name:</label>
              <input
                type="text"
                name="firstName"
                value={user.firstName || ''}
                onChange={this.handleFieldChange}
              />
            </div>
            <div>
              <label>Last Name:</label>
              <input
                type="text"
                name="lastName"
                value={user.lastName || ''}
                onChange={this.handleFieldChange}
              />
            </div>
            <div>
              <button type="submit" disabled={isLoading}>Save</button>
              <button type="button" onClick={this.handleCancel}>Cancel</button>
            </div>
          </form>
        ) : (
          <div className="user-info">
            <p><strong>Username:</strong> {user.username}</p>
            <p><strong>Email:</strong> {user.email}</p>
            <p><strong>Name:</strong> {`${user.firstName || ''} ${user.lastName || ''}`.trim() || 'N/A'}</p>
            <p><strong>Status:</strong> {user.isActive ? 'Active' : 'Inactive'}</p>
            <p><strong>Roles:</strong> {user.roles.join(', ')}</p>
            
            <button onClick={this.handleEdit}>Edit</button>
          </div>
        )}
        
        {showAdminControls && this.renderAdminControls()}
      </div>
    );
  }
}

/**
 * Functional component version of UserProfile using hooks
 */
export const UserProfileHooks: React.FC<UserProfileProps> = (props) => {
  const { userId, data, isLoading: propsLoading, showAdminControls, onUserUpdate, className, style } = props;
  
  // State hooks
  const [isEditing, setIsEditing] = useState(false);
  const [user, setUser] = useState<User | null>(data || null);
  const [isLoading, setIsLoading] = useState(propsLoading || false);
  const [error, setError] = useState<string | null>(null);
  
  // Effect to update user when props change
  useEffect(() => {
    if (data) {
      setUser(data);
    }
  }, [data]);
  
  // Effect to update loading state when props change
  useEffect(() => {
    setIsLoading(propsLoading || false);
  }, [propsLoading]);
  
  // Check if user is admin using memoization
  const isAdmin = useMemo(() => {
    if (!user) return false;
    return user.roles.includes('admin') && 'permissions' in user;
  }, [user]);
  
  // Handle edit button click
  const handleEdit = useCallback(() => {
    setIsEditing(true);
  }, []);
  
  // Handle save button click
  const handleSave = useCallback(() => {
    if (!user) return;
    
    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      setIsEditing(false);
      
      if (onUserUpdate) {
        onUserUpdate(user);
      }
    }, 1000);
  }, [user, onUserUpdate]);
  
  // Handle cancel button click
  const handleCancel = useCallback(() => {
    setIsEditing(false);
    setUser(data || null);
  }, [data]);
  
  // Handle field changes
  const handleFieldChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
    
    setUser(prevUser => prevUser ? {
      ...prevUser,
      [name]: value
    } : null);
  }, []);
  
  // Render component
  if (isLoading) {
    return <div className="loading">Loading user data...</div>;
  }
  
  if (error) {
    return <div className="error">{error}</div>;
  }
  
  if (!user) {
    return <div className="no-data">No user data available</div>;
  }
  
  return (
    <div className={`user-profile ${className || ''}`} style={style}>
      <h2>User Profile (Hooks)</h2>
      
      {isEditing ? (
        <form onSubmit={e => { e.preventDefault(); handleSave(); }}>
          <div>
            <label>Username:</label>
            <input
              type="text"
              name="username"
              value={user.username}
              onChange={handleFieldChange}
              disabled
            />
          </div>
          <div>
            <label>Email:</label>
            <input
              type="email"
              name="email"
              value={user.email}
              onChange={handleFieldChange}
            />
          </div>
          <div>
            <label>First Name:</label>
            <input
              type="text"
              name="firstName"
              value={user.firstName || ''}
              onChange={handleFieldChange}
            />
          </div>
          <div>
            <label>Last Name:</label>
            <input
              type="text"
              name="lastName"
              value={user.lastName || ''}
              onChange={handleFieldChange}
            />
          </div>
          <div>
            <button type="submit" disabled={isLoading}>Save</button>
            <button type="button" onClick={handleCancel}>Cancel</button>
          </div>
        </form>
      ) : (
        <div className="user-info">
          <p><strong>Username:</strong> {user.username}</p>
          <p><strong>Email:</strong> {user.email}</p>
          <p><strong>Name:</strong> {`${user.firstName || ''} ${user.lastName || ''}`.trim() || 'N/A'}</p>
          <p><strong>Status:</strong> {user.isActive ? 'Active' : 'Inactive'}</p>
          <p><strong>Roles:</strong> {user.roles.join(', ')}</p>
          
          <button onClick={handleEdit}>Edit</button>
        </div>
      )}
      
      {showAdminControls && isAdmin && (
        <div className="admin-controls">
          <h3>Admin Controls</h3>
          <div>
            <label>Permissions:</label>
            <ul>
              {(user as AdminUser).permissions.map(perm => (
                <li key={perm}>{perm}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

// Export authenticated versions of components
export const AuthenticatedUserProfile = withAuthentication(UserProfile);
export const AuthenticatedUserProfileHooks = withAuthentication(UserProfileHooks);