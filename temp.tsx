
import React from 'react';

interface Props {
    name: string;
    age: number;
}

export class Person extends React.Component<Props> {
    render() {
        const { name, age } = this.props;
        return (
            <div>
                <h1>Hello, {name}!</h1>
                <p>You are {age} years old.</p>
            </div>
        );
    }
}
