public class Archer extends Character {
    // Constructor
    public Archer(String name) {
        super(name, CharacterStats.ARCHER);
    }

    // Special Archer attack
    @Override
    public void specialAttack(Character defender) {
        System.out.println("You are using a magical arrow.");
        attack(defender);
    }
}
