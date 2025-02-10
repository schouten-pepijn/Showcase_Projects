public class Warrior extends Character {
    public Warrior(String name) {
        super(name, CharacterStats.WARRIOR);
    }

    @Override
    public void specialAttack(Character defender) {
        System.out.println("You are swining your giant hammer.");
        attack(defender);
    }
}
