import streamlit as st
import random
import time

# Page configuration
st.set_page_config(
    page_title="Rock Paper Scissors Championship",
    page_icon="✂️",
    layout="centered"
)

# Initialize session state
if 'player_score' not in st.session_state:
    st.session_state.player_score = 0
if 'computer_score' not in st.session_state:
    st.session_state.computer_score = 0
if 'rounds_played' not in st.session_state:
    st.session_state.rounds_played = 0
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
if 'last_round_result' not in st.session_state:
    st.session_state.last_round_result = ""
if 'game_history' not in st.session_state:
    st.session_state.game_history = []

# Game constants
CHOICES = ['rock', 'paper', 'scissors']
EMOJIS = {'rock': '🪨', 'paper': '📄', 'scissors': '✂️'}
WINNING_MOVES = {
    'rock': 'scissors',
    'scissors': 'paper', 
    'paper': 'rock'
}

def reset_game():
    """Reset game state"""
    st.session_state.player_score = 0
    st.session_state.computer_score = 0
    st.session_state.rounds_played = 0
    st.session_state.game_over = False
    st.session_state.last_round_result = ""
    st.session_state.game_history = []

def play_round(player_choice):
    """Play one round of the game"""
    computer_choice = random.choice(CHOICES)
    st.session_state.rounds_played += 1
    
    # Determine winner
    if player_choice == computer_choice:
        result = "tie"
        result_text = "⚖️ It's a TIE!"
    elif WINNING_MOVES[player_choice] == computer_choice:
        result = "player"
        result_text = "✨ You WIN this round!"
        st.session_state.player_score += 1
    else:
        result = "computer"
        result_text = "💔 Computer wins this round!"
        st.session_state.computer_score += 1
    
    # Store round info
    round_info = {
        'round': st.session_state.rounds_played,
        'player_choice': player_choice,
        'computer_choice': computer_choice,
        'result': result,
        'player_score': st.session_state.player_score,
        'computer_score': st.session_state.computer_score
    }
    st.session_state.game_history.append(round_info)
    
    # Update last round result for display
    st.session_state.last_round_result = f"""
    **Round {st.session_state.rounds_played}**
    
    You chose: {EMOJIS[player_choice]} **{player_choice.upper()}**
    
    Computer chose: {EMOJIS[computer_choice]} **{computer_choice.upper()}**
    
    {result_text}
    """
    
    # Check if game is over
    if st.session_state.player_score == 3 or st.session_state.computer_score == 3:
        st.session_state.game_over = True

# Main app
st.title("🎮 ROCK, PAPER, SCISSORS CHAMPIONSHIP! 🎮")
st.markdown("---")
st.markdown("**First to 3 wins takes the championship!**")

# Score display
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.metric("Your Score", st.session_state.player_score)
with col2:
    st.markdown(f"### Round {st.session_state.rounds_played + 1}")
with col3:
    st.metric("Computer Score", st.session_state.computer_score)

# Game over check
if st.session_state.game_over:
    st.markdown("---")
    if st.session_state.player_score == 3:
        st.success("🏆 CONGRATULATIONS! YOU ARE THE CHAMPION! 🏆")
        st.balloons()
    else:
        st.error("🤖 The computer wins this time. Try again!")
    
    st.info(f"Final Score: You {st.session_state.player_score} - {st.session_state.computer_score} Computer")
    st.info(f"Total rounds played: {st.session_state.rounds_played}")
    
    if st.button("🔄 Play Again", type="primary"):
        reset_game()
        st.rerun()

# Game controls (only show if game not over)
elif not st.session_state.game_over:
    st.markdown("### Choose your weapon:")
    
    # Create buttons for choices
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(f"🪨\n**Rock**", use_container_width=True, key="rock_btn"):
            play_round('rock')
            st.rerun()
    
    with col2:
        if st.button(f"📄\n**Paper**", use_container_width=True, key="paper_btn"):
            play_round('paper')
            st.rerun()
    
    with col3:
        if st.button(f"✂️\n**Scissors**", use_container_width=True, key="scissors_btn"):
            play_round('scissors')
            st.rerun()

# Show last round result
if st.session_state.last_round_result:
    st.markdown("---")
    st.markdown("### Last Round:")
    st.markdown(st.session_state.last_round_result)

# Game statistics (expandable section)
if st.session_state.rounds_played > 0:
    with st.expander("📊 Game Statistics"):
        # Calculate statistics
        player_wins = len([r for r in st.session_state.game_history if r['result'] == 'player'])
        computer_wins = len([r for r in st.session_state.game_history if r['result'] == 'computer'])
        ties = len([r for r in st.session_state.game_history if r['result'] == 'tie'])
        
        # Display stats
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.metric("Your Wins", player_wins)
        with stats_col2:
            st.metric("Computer Wins", computer_wins)
        with stats_col3:
            st.metric("Ties", ties)
        
        # Choice frequency
        if st.session_state.game_history:
            st.markdown("**Your Choice Frequency:**")
            choice_counts = {}
            for choice in CHOICES:
                count = len([r for r in st.session_state.game_history if r['player_choice'] == choice])
                choice_counts[choice] = count
                if count > 0:
                    st.write(f"{EMOJIS[choice]} {choice.title()}: {count} times")

# Game history (expandable section)
if st.session_state.game_history:
    with st.expander("📋 Round History"):
        for round_info in reversed(st.session_state.game_history):
            result_emoji = {"player": "✅", "computer": "❌", "tie": "⚖️"}
            st.write(f"**Round {round_info['round']}:** {EMOJIS[round_info['player_choice']]} vs {EMOJIS[round_info['computer_choice']]} {result_emoji[round_info['result']]}")

# Game rules (expandable section)
with st.expander("📖 How to Play"):
    st.markdown("""
    **Rules:**
    - Rock 🪨 crushes Scissors ✂️
    - Scissors ✂️ cuts Paper 📄  
    - Paper 📄 covers Rock 🪨
    - First player to win 3 rounds wins the championship!
    
    **Tips:**
    - Try to be unpredictable in your choices
    - Look for patterns in the computer's play (though it's random!)
    - Have fun! 🎉
    """)

# Reset button (always available)
st.markdown("---")
if st.button("🔄 Reset Game"):
    reset_game()
    st.rerun()

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit | Converted from Jupyter Notebook*")