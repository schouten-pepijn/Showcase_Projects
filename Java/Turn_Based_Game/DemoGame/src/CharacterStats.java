public enum CharacterStats {
    WARRIOR(
        120, 30, 40,
        6, 30, 30,
        10
    ),
    MAGE(
        90, 15, 40,
        2, 10, 30,
        10
    ),
    ARCHER(
        110, 20, 30,
        4, 20, 40,
        20
    );

    private final int health;
    private final int attackPower;
    private final int specialAttackPower;
    private final int defencePower;
    private final int specialAttackCost;
    private final int beginMana;
    private final int manaIncrease;

    private CharacterStats(
        int health, int attackPower, int specialAttackPower,
        int defencePower, int specialAttackCost, int beginMana,
        int manaIncrease
    ) {
        this.health = health;
        this.attackPower = attackPower;
        this.specialAttackPower = specialAttackPower;
        this.defencePower = defencePower;
        this.specialAttackCost = specialAttackCost;
        this.beginMana = beginMana;
        this.manaIncrease = manaIncrease;
    }

    public int getHealth() {
        return health;
    }

    public int getAttackPower() {
        return attackPower;
    }

    public int getSpecialAttackPower() {
        return specialAttackPower;
    }

    public int getDefencePower() {
        return defencePower;
    }

    public int getSpecialAttackCost() {
        return specialAttackCost;
    }

    public int getBeginMana() {
        return beginMana;
    }

    public int getManaIncrease() {
        return manaIncrease;
    }
    
}




