/**
 * Complex TypeScript interface hierarchy for testing AST extraction
 */

/**
 * Base entity interface with common properties
 */
export interface Entity {
  id: string;
  createdAt: Date;
  updatedAt: Date;
}

/**
 * Metadata interface for items that have additional properties
 */
export interface Metadata {
  [key: string]: string | number | boolean | null;
}

/**
 * Auditable interface for tracking changes
 */
export interface Auditable {
  createdBy?: string;
  updatedBy?: string;
  version: number;
}

/**
 * Represents something that can be serialized to JSON
 */
export interface Serializable<T> {
  toJSON(): Record<string, any>;
  fromJSON(data: Record<string, any>): T;
}

/**
 * Validator interface for data validation
 */
export interface Validator<T> {
  validate(data: T): boolean;
  getErrors(): string[];
}

/**
 * User model interface extending multiple interfaces
 */
export interface User extends Entity, Auditable {
  username: string;
  email: string;
  firstName?: string;
  lastName?: string;
  isActive: boolean;
  roles: string[];
  metadata?: Metadata;
}

/**
 * Admin user with additional privileges
 */
export interface AdminUser extends User {
  permissions: string[];
  canImpersonate: boolean;
}

/**
 * Represents a product in the system
 */
export interface Product extends Entity, Serializable<Product> {
  name: string;
  description: string;
  price: number;
  categories: string[];
  tags: string[];
  isAvailable: boolean;
  metadata?: Metadata;
}

/**
 * Subscription interface with complex structure
 */
export interface Subscription<T extends User> extends Entity, Auditable {
  user: T;
  plan: string;
  startDate: Date;
  endDate?: Date;
  isActive: boolean;
  paymentHistory: Payment[];
  settings: SubscriptionSettings;
}

/**
 * Payment record interface
 */
export interface Payment {
  id: string;
  amount: number;
  currency: string;
  date: Date;
  method: PaymentMethod;
  status: PaymentStatus;
}

/**
 * Enum for payment methods
 */
export enum PaymentMethod {
  CreditCard = 'credit_card',
  BankTransfer = 'bank_transfer',
  PayPal = 'paypal',
  Crypto = 'crypto'
}

/**
 * Enum for payment status
 */
export enum PaymentStatus {
  Pending = 'pending',
  Completed = 'completed',
  Failed = 'failed',
  Refunded = 'refunded'
}

/**
 * Subscription settings interface
 */
export interface SubscriptionSettings {
  autoRenew: boolean;
  notifications: NotificationSettings;
  features: Record<string, boolean>;
}

/**
 * Notification settings interface
 */
export interface NotificationSettings {
  email: boolean;
  sms: boolean;
  pushNotifications: boolean;
  reminderDays: number;
}

/**
 * Type for API responses
 */
export type ApiResponse<T> = {
  success: boolean;
  data?: T;
  error?: string;
  meta?: {
    page?: number;
    total?: number;
    limit?: number;
  };
};

/**
 * Component props interface with generic type
 */
export interface ComponentProps<T = any> {
  data?: T;
  isLoading?: boolean;
  error?: Error | null;
  onAction?: (action: string, payload?: any) => void;
  className?: string;
  style?: React.CSSProperties;
  children?: React.ReactNode;
}