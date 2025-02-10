import java.util.Random;
import java.util.Scanner;

public class Battle implements Potionable {
    private Character attacker;
    private Character defender;
    private Scanner scanner = new Scanner(System.in);
    private Random random = new Random();

    // Constructor
    public Battle(Character attacker, Character defender) {
        this.attacker = attacker;
        this.defender = defender;
    }

    // Start of the battle
    public void start() {
        whoStarts();  // Random who starts
        attackLoop();  // battle Loop
        printWinner();  // Select who won
        scanner.close();
    }

    // Random Potion 20% chance
    @Override
    public void randomPotion() {
        if (random.nextFloat() < potionChance) {
            defender.setHealth(defender.getHealth() + potionHeal);
            System.out.println("You found a health potion!");
            System.out.printf("New health: %d!%n", defender.getHealth());
        }
    }
    
    // Random who starts
    private void whoStarts() {
        if (random.nextBoolean())
            swapTurn();
    }

    // Swap turns
    private void swapTurn() {
        Character temp = attacker;
        attacker = defender;
        defender = temp;
    }

    // Battle loop
    private void attackLoop() {
        // check if a character died
        while (attacker.stillAlive() && defender.stillAlive()) {
            attackStep();  // Do an attack
            randomPotion();  // Random Potion?
        }
    }

    // Determine who won
    private void printWinner() {
        System.out.println("Battle over!");
        System.out.printf(
            "%s wins!%n", attacker.stillAlive() ? attacker.getName() : defender.getName()
        );
    }

    // Select if you want normal attack, special of check healths
    private void attackStep() {
        System.out.printf("%n%s's turn.%n", attacker.getName());
        System.out.println("(1) Normal Attack (2) Special Attack (3) Check Defenders Health");
        System.out.print("What will you do? ");
        int choice = scanner.nextInt();

        switch (choice) {
            case 1:
                attacker.setNormalAttack(true);
                attacker.attack(defender);
                swapTurn();
                break;
            case 2:
                attacker.setNormalAttack(false);
                attacker.specialAttack(defender);
                swapTurn();
                break;
            case 3:
                System.out.printf(
                    "%s has %d health left.%n", defender.getName(), defender.getHealth());
                break;
            default:
                System.out.println("Invalid choice. Select 1, 2 or 3.");
                break;
        }
    }
}
