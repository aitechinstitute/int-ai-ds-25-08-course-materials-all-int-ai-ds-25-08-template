# 🎮 Rock Paper Scissors Championship

A fun, interactive Rock Paper Scissors game built with Streamlit! This web app was converted from a Jupyter notebook into a full-featured game with real-time scoring, statistics tracking, and a beautiful user interface.

## 🎯 Features

- **Interactive Gameplay**: Click buttons to make your choice - no typing required!
- **Championship Mode**: First to 3 wins takes the championship
- **Real-time Scoring**: Live score tracking with visual metrics
- **Game Statistics**: Track your win rate and choice patterns
- **Round History**: Complete log of all rounds played
- **Visual Feedback**: Emojis, animations, and clear result displays
- **Mobile Friendly**: Responsive design that works on all devices

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation & Running

1. **Clone or Download** this repository to your computer

2. **Navigate to the project folder** in your terminal/command prompt:
   ```bash
   cd rock-paper-scissors-app
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser** to `http://localhost:8501` (usually opens automatically)

## 🎲 How to Play

1. **Choose your weapon**: Click Rock 🪨, Paper 📄, or Scissors ✂️
2. **Battle the computer**: See who wins each round
3. **First to 3 wins**: Takes the championship!
4. **Track your progress**: View statistics and game history

### Game Rules
- Rock 🪨 crushes Scissors ✂️
- Scissors ✂️ cuts Paper 📄
- Paper 📄 covers Rock 🪨

## 📁 Project Structure

```
rock-paper-scissors-app/
├── streamlit_app.py      # Main application code
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Technical Details

### Built With
- **Streamlit**: Web app framework
- **Python**: Programming language
- **Random**: For computer choice generation

### Key Components
- **Session State Management**: Maintains game state across interactions
- **Interactive UI**: Button-based gameplay
- **Statistics Tracking**: Win rates and choice analysis
- **Responsive Design**: Clean, mobile-friendly layout

## 🌐 Deployment

### Deploy to Streamlit Community Cloud (Free)

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Visit [share.streamlit.io](https://share.streamlit.io)**

3. **Connect your GitHub repository**

4. **Deploy with one click!**

### Other Deployment Options
- **Railway**: Connect GitHub repo and deploy
- **Render**: Similar GitHub integration
- **Heroku**: Add `Procfile` with: `web: streamlit run streamlit_app.py --server.port=$PORT`

## 🎨 Screenshots

*Game in action:*
- Clean, intuitive interface with emoji buttons
- Real-time score tracking
- Statistics and history panels
- Victory celebrations with balloons! 🎉

## 🔄 Version History

### v1.0.0 (Current)
- ✅ Full Rock Paper Scissors gameplay
- ✅ Championship mode (first to 3 wins)
- ✅ Real-time scoring and statistics
- ✅ Round history tracking
- ✅ Mobile-responsive design
- ✅ Converted from Jupyter notebook

## 🤝 Contributing

Want to improve the game? Great! Here are some ideas:

### Enhancement Ideas
- [ ] Add sound effects
- [ ] Multiple difficulty levels
- [ ] Player vs Player mode
- [ ] Tournament brackets
- [ ] Different game variations (Rock Paper Scissors Lizard Spock)
- [ ] Save game history to file
- [ ] Leaderboards

### How to Contribute
1. Fork this repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

**App won't start:**
```bash
# Try using python -m streamlit instead
python -m streamlit run streamlit_app.py
```

**Port already in use:**
- Streamlit will automatically use the next available port (8502, 8503, etc.)
- Or specify a port: `streamlit run streamlit_app.py --server.port 8080`

**Dependencies not installing:**
```bash
# Try upgrading pip first
pip install --upgrade pip
pip install -r requirements.txt
```

## 📞 Support

Having issues? Here's how to get help:

1. **Check the troubleshooting section** above
2. **Create an issue** on this repository with:
   - Your operating system
   - Python version (`python --version`)
   - Error message (if any)
   - Steps to reproduce the problem

## 📝 License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).

## 🙏 Acknowledgments

- Original game concept converted from Jupyter notebook
- Built with the amazing [Streamlit](https://streamlit.io) framework
- Emojis make everything better! 🎉

---

**Ready to play?** Run the app and challenge the computer to a championship match! 🏆

*Made with ❤️ and Python*