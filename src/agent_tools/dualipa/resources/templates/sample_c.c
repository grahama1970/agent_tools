/**
 * Sample C file with various functions, structs, and directives for testing
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NAME_LENGTH 100
#define PI 3.14159
#define SQUARE(x) ((x) * (x))

/**
 * Simple Person struct to store name and age
 */
typedef struct {
    char name[MAX_NAME_LENGTH];
    int age;
} Person;

/**
 * Greet function that takes a name and prints a greeting
 */
void greet(const char* name) {
    printf("Hello, %s!\n", name);
}

/**
 * Create a new Person structure
 */
Person* create_person(const char* name, int age) {
    Person* p = (Person*)malloc(sizeof(Person));
    if (p == NULL) {
        return NULL;
    }
    
    strncpy(p->name, name, MAX_NAME_LENGTH - 1);
    p->name[MAX_NAME_LENGTH - 1] = '\0';
    p->age = age;
    
    return p;
}

/**
 * Calculate the area of a circle
 */
double circle_area(double radius) {
    return PI * SQUARE(radius);
}

/**
 * Main function that demonstrates usage
 */
int main(int argc, char** argv) {
    // Create a person
    Person* john = create_person("John Doe", 30);
    
    // Greet the person
    greet(john->name);
    
    // Calculate and print a circle area
    double area = circle_area(5.0);
    printf("Area of circle: %.2f\n", area);
    
    // Clean up
    free(john);
    
    return 0;
} 