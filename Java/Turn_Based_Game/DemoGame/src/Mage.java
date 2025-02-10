public class Mage extends Character {
    public Mage(String name) {
        super(name, CharacterStats.MAGE);
    }

    @Override
    public void specialAttack(Character defender) {
        System.out.println("You are casting a fireball spell.");
        attack(defender);
    }
}
