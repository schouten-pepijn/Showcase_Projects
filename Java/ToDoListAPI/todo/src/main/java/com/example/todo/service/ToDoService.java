package com.example.todo.service;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.example.todo.model.ToDo;
import com.example.todo.repository.ToDoRepository;

// Class for business logic
@Service
public class ToDoService {

    // Dependency injection
    @Autowired
    private ToDoRepository repository;

    // Get all ToDos
    public List<ToDo> getAllToDos() {
        return repository.findAll();
    }

    // Get a Task by ID
    public ToDo getToDoById(Long id) {
        return repository.findById(id).orElse(null);
    }

    // Save a Task
    public ToDo saveToDo(ToDo toDo) {
        return repository.save(toDo);
    }

    // Delete a Task by ID
    public void deleteToDo(Long id) {
        repository.deleteById(id);
    }

}
