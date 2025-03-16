// Sample JavaScript file for testing code extraction
import { useState, useEffect } from 'react';
import axios from 'axios';

/**
 * A function to fetch data from an API
 * @param {string} url - The URL to fetch data from
 * @param {Object} options - Optional configuration
 * @returns {Promise<Object>} The response data
 */
async function fetchData(url, options = {}) {
    try {
        const response = await axios.get(url, options);
        return response.data;
    } catch (error) {
        console.error('Error fetching data:', error);
        throw error;
    }
}

/**
 * User class representing a user in the system
 */
class User {
    /**
     * Create a user
     * @param {string} name - The user's name
     * @param {string} email - The user's email
     */
    constructor(name, email) {
        this.name = name;
        this.email = email;
        this.createdAt = new Date();
    }

    /**
     * Get user's full profile
     * @returns {Object} User profile
     */
    getProfile() {
        return {
            name: this.name,
            email: this.email,
            createdAt: this.createdAt
        };
    }

    /**
     * Update user's name
     * @param {string} newName - New name for the user
     */
    updateName(newName) {
        this.name = newName;
    }
}

/**
 * Admin class extending User with additional privileges
 */
class Admin extends User {
    /**
     * Create an admin
     * @param {string} name - Admin name
     * @param {string} email - Admin email
     * @param {string[]} permissions - Admin permissions
     */
    constructor(name, email, permissions = []) {
        super(name, email);
        this.permissions = permissions;
    }

    /**
     * Check if admin has a specific permission
     * @param {string} permission - Permission to check
     * @returns {boolean} Whether admin has the permission
     */
    hasPermission(permission) {
        return this.permissions.includes(permission);
    }
}

/**
 * React hook for managing API data
 * @param {string} url - API endpoint
 * @returns {Object} Data and loading state
 */
const useApiData = (url) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const loadData = async () => {
            try {
                setLoading(true);
                const result = await fetchData(url);
                setData(result);
            } catch (err) {
                setError(err);
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
 * @param {Date} date - Date object to format
 * @param {string} format - Format string
 * @returns {string} Formatted date string
 */
const formatDate = (date, format = 'short') => {
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

// Default export
export default User;

// Named exports
export { fetchData, useApiData, formatDate, Admin }; 