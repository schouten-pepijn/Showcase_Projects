import java.util.Scanner;

public class Player implements Playerable {
    private Scanner scanner = new Scanner(System.in);
    private int choice;
    private String name;

    // Choose character
    @Override
    public Character chooseCharacter() {
        userInput();
        return CharacterFactory();
    }
    
    // Get user input from terminal
    @Override
    public void userInput() {
        System.out.println("Welcome to the game!");
        System.out.println("Please choose your character..");
        System.out.print("(1) Warrior (2) Mage (3) Archer: ");
        choice = scanner.nextInt();

        System.out.println("What is your player name? ");
        scanner.nextLine();
        name = scanner.nextLine();
    }

    // Select the character
    @Override
    public Character CharacterFactory() {
        switch (choice) {
            case 1:
                return new Warrior(name);
            case 2:
                return new Mage(name);
            case 3:
                return new Archer(name);
            default:
                System.out.println("Invalid choice. Default to Warrior.");
                return new Warrior(name);
        }
    }
}
