public class StartGame {
    public static void main(String[] args) {

        Player player1 = new Player();
        Player player2 = new Player();

        Character character1 = player1.chooseCharacter();
        Character character2 = player2.chooseCharacter();

        Battle battle = new Battle(character1, character2);
        battle.start();
    }
}
