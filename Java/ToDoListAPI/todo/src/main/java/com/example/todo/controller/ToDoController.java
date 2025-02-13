package com.example.todo.controller;

import java.util.List;

import javax.validation.Valid;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.todo.model.ToDo;
import com.example.todo.service.ToDoService;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.PutMapping;

// Handle HTTP requests on "/todos"
@RestController
@RequestMapping("/todos")
public class ToDoController {

    @Autowired
    private ToDoService service;

    // Get all ToDos
    @GetMapping
    public List<ToDo> getAll() {
        return service.getAllToDos();
    }

    // Get a Task by ID ("/{id}" endpoint)
    @GetMapping("/{id}")
    // Injects the value of the "id" parameter into the method
    public ResponseEntity<ToDo> getById(@PathVariable Long id) {
        ToDo toDo = service.getToDoById(id);
        // If the ToDo is found, return it as a ResponseEntity else return 404
        return toDo != null ? ResponseEntity.ok(toDo) : ResponseEntity.notFound().build();
    }

    // Create a Task
    @PostMapping()
    // Injects the request body into the method (Validation)
    public ToDo create(@Valid @RequestBody ToDo toDo) {
        return service.saveToDo(toDo);
    }

    // Update a Task ("/{id}" endpoint)
    @PutMapping("/{id}")
    // Injects the value of the "id" parameter into the method
    // Injects the request body into the method
    public ResponseEntity<ToDo> update(@PathVariable Long id, @Valid @RequestBody ToDo updatedToDo) {
        // Check if the ToDo exists
        ToDo existingToDo = service.getToDoById(id);
        if (existingToDo == null)
            return ResponseEntity.notFound().build();

        // Update the ToDo
        existingToDo.setTitle(updatedToDo.getTitle());
        existingToDo.setCompleted(updatedToDo.isCompleted());

        // Save the updated ToDo
        return ResponseEntity.ok(service.saveToDo(existingToDo));
    }

    // Delete a Task ("/{id}" endpoint)
    @DeleteMapping("/{id}")
    // Injects the value of the "id" parameter into the method
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        // Delete the ToDo
        service.deleteToDo(id);
        // Return a 204 response
        return ResponseEntity.noContent().build();
    }

}
