import java.util.Random;

public abstract class Character {
    private String name;
    private int health;
    private int attackPower;
    private int specialAttackPower;
    private int defencePower;
    private int specialAttackCost;
    private int beginMana;
    private int manaIncrease;
    private boolean normalAttack = true;
    private Random random = new Random();

    // Constructor
    public Character(String name, CharacterStats characterStats) 
    {
        this.name = name;
        this.health = characterStats.getHealth();
        this.attackPower = characterStats.getAttackPower();
        this.specialAttackPower = characterStats.getSpecialAttackPower();
        this.defencePower = characterStats.getDefencePower();
        this.specialAttackCost = characterStats.getSpecialAttackCost();
        this.beginMana = characterStats.getBeginMana(); 
        this.manaIncrease = characterStats.getManaIncrease();
    }

    // Attack method of the character
    public void attack(Character character) {
        System.out.printf("%s is attacking...%n", this.name);

        // normal attack
        if (normalAttack) {
            character.takeDamage(attackPower);
            increaseMana();
            return;
        }

        // special attack
        if (checkMana()) {
            character.takeDamage(specialAttackPower);
            reduceMana();
        }
    }

    // Getters and setters
    public boolean stillAlive() {
        return health > 0;
    }

    public String getName() {
        return name;
    }

    public int getHealth() {
        return health;
    }

    public void setNormalAttack(boolean normalAttack) {
        this.normalAttack = normalAttack;
    }

    public void setHealth(int health) {
        this.health = health;
    }

    // abstract methods for specific implementation
    public abstract void specialAttack(Character character);

    // Print health on screen
    private void printHealth(Character character) {
        System.out.printf(
            "%s has %d health left.%n", character.getName(), character.getHealth()
        );
    }

    // Increase the mana
    private void increaseMana() {
        beginMana += manaIncrease;
        System.out.printf("%nMana increased to %d.%n", beginMana);
    }

    // Check the mana
    private boolean checkMana() {
        if (beginMana < specialAttackCost) {
            System.out.println("Out of mana! No Damage done.");
            return false;
        }
        return true;
    }

    // Reduce the mana
    private void reduceMana() {
        beginMana = Math.max(0, beginMana - specialAttackCost);
        System.out.printf("%nMana reduced to %d.%n", beginMana);
    }

    // Random damage modifier
    private int randomDamage() {
        return random.nextInt(7) - 3;
    }

    // Take damage 
    private void takeDamage(int damage) {
        int finalDamage = health + defencePower - damage - randomDamage();
        health = Math.max(0, finalDamage);
        printHealth(this);
    }
}
