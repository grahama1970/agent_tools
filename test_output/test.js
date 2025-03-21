
function greet(name) {
    console.log(`Hello, ${name}!`);
}

class Person {
    constructor(name) {
        this.name = name;
    }
    
    sayHello() {
        console.log(`Hi, I'm ${this.name}`);
    }
}
