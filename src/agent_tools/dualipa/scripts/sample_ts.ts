// Sample TypeScript file for testing code extraction
import { useState, useEffect } from 'react';
import axios, { AxiosRequestConfig, AxiosResponse } from 'axios';

/**
 * User data interface
 */
interface UserData {
    id: number;
    name: string;
    email: string;
    role: 'user' | 'admin';
    createdAt: Date;
}

/**
 * API response type
 */
type ApiResponse<T> = {
    data: T;
    status: number;
    message: string;
    timestamp: Date;
};

/**
 * Configuration options for API requests
 */
interface ApiOptions extends AxiosRequestConfig {
    cache?: boolean;
    timeout?: number;
}

/**
 * A function to fetch data from an API
 * @param url - The URL to fetch data from
 * @param options - Optional configuration
 * @returns Promise with the response data
 */
async function fetchData<T>(url: string, options: ApiOptions = {}): Promise<ApiResponse<T>> {
    try {
        const response: AxiosResponse = await axios.get(url, options);
        return {
            data: response.data,
            status: response.status,
            message: 'Success',
            timestamp: new Date()
        };
    } catch (error) {
        console.error('Error fetching data:', error);
        throw error;
    }
}

/**
 * User class representing a user in the system
 */
class User implements UserData {
    id: number;
    name: string;
    email: string;
    role: 'user' | 'admin';
    createdAt: Date;

    /**
     * Create a user
     * @param name - The user's name
     * @param email - The user's email
     * @param id - Optional user ID
     */
    constructor(name: string, email: string, id?: number) {
        this.name = name;
        this.email = email;
        this.id = id || Math.floor(Math.random() * 10000);
        this.role = 'user';
        this.createdAt = new Date();
    }

    /**
     * Get user's full profile
     * @returns User profile object
     */
    getProfile(): UserData {
        return {
            id: this.id,
            name: this.name,
            email: this.email,
            role: this.role,
            createdAt: this.createdAt
        };
    }

    /**
     * Update user's name
     * @param newName - New name for the user
     */
    updateName(newName: string): void {
        this.name = newName;
    }
}

/**
 * Admin class extending User with additional privileges
 */
class Admin extends User {
    permissions: string[];

    /**
     * Create an admin
     * @param name - Admin name
     * @param email - Admin email
     * @param permissions - Admin permissions
     */
    constructor(name: string, email: string, permissions: string[] = []) {
        super(name, email);
        this.role = 'admin';
        this.permissions = permissions;
    }

    /**
     * Check if admin has a specific permission
     * @param permission - Permission to check
     * @returns Whether admin has the permission
     */
    hasPermission(permission: string): boolean {
        return this.permissions.includes(permission);
    }
}

/**
 * React hook for managing API data
 * @param url - API endpoint
 * @returns Object with data and loading state
 */
const useApiData = <T>(url: string) => {
    const [data, setData] = useState<T | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<Error | null>(null);

    useEffect(() => {
        const loadData = async () => {
            try {
                setLoading(true);
                const result = await fetchData<T>(url);
                setData(result.data);
            } catch (err) {
                setError(err as Error);
            } finally {
                setLoading(false);
            }
        };

        loadData();
    }, [url]);

    return { data, loading, error };
};

/**
 * Format a date string
 * @param date - Date object to format
 * @param format - Format string
 * @returns Formatted date string
 */
const formatDate = (date?: Date, format: 'short' | 'long' | 'default' = 'short'): string => {
    if (!date) return '';
    
    switch (format) {
        case 'short':
            return date.toLocaleDateString();
        case 'long':
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
        default:
            return date.toString();
    }
};

// Utility type
type Nullable<T> = T | null;

// Generic function example
function getFirstItem<T>(items: T[]): Nullable<T> {
    return items.length > 0 ? items[0] : null;
}

// Default export
export default User;

// Named exports
export { fetchData, useApiData, formatDate, Admin };
export type { UserData, ApiResponse, ApiOptions, Nullable }; 