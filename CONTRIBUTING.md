# Contributing to FPL AI Assistant

Thank you for your interest in contributing to the FPL AI Assistant! 🎉

## 🚀 Ways to Contribute

### 🐛 Bug Reports
- Use the [issue tracker](https://github.com/YOUR_USERNAME/fpl-ai-assistant/issues)
- Include steps to reproduce the bug
- Mention your Python version and OS

### 💡 Feature Requests
- Check existing issues first
- Describe the feature and its benefits
- Consider the FPL use case

### 🔧 Code Contributions
- Fork the repository
- Create a feature branch
- Make your changes
- Add tests if applicable
- Submit a pull request

## 🛠️ Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/fpl-ai-assistant.git
cd fpl-ai-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## 📝 Code Style

### Python Code Style
- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting
- Maximum line length: 88 characters
- Use type hints where appropriate

### Documentation
- Add docstrings to all functions and classes
- Use Google-style docstrings
- Update README.md for new features

### Example Function
```python
def predict_player_points(player_data: pd.DataFrame, gameweek: int) -> float:
    """Predict FPL points for a player in a specific gameweek.
    
    Args:
        player_data: DataFrame containing player statistics
        gameweek: The gameweek number to predict for
        
    Returns:
        Predicted FPL points for the player
        
    Raises:
        ValueError: If player_data is empty or gameweek is invalid
    """
    # Implementation here
    pass
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_predictor.py
```

### Writing Tests
- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external APIs (FPL API calls)

## 📋 Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes
4. **Add** tests for new functionality
5. **Run** tests: `pytest`
6. **Format** code: `black .`
7. **Lint** code: `flake8`
8. **Commit** changes: `git commit -m 'Add amazing feature'`
9. **Push** to branch: `git push origin feature/amazing-feature`
10. **Submit** a Pull Request

### Pull Request Guidelines
- Reference any related issues
- Describe what your PR does
- Include screenshots for UI changes
- Ensure all tests pass
- Keep PR focused and atomic

## 🎯 Priority Areas

We especially welcome contributions in these areas:

### 🤖 Machine Learning
- Alternative prediction models (Neural Networks, Random Forest)
- Feature engineering improvements
- Model performance optimization
- Cross-validation strategies

### 📊 Data Analysis
- New data sources integration
- Advanced FPL metrics
- Historical analysis features
- Price change prediction

### 🎨 User Interface
- Streamlit component improvements
- Mobile responsiveness
- New visualization charts
- User experience enhancements

### 🔧 Infrastructure
- CI/CD pipeline setup
- Docker containerization
- Performance optimizations
- Error handling improvements

## 🏷️ Issue Labels

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention is needed
- `ml` - Machine learning related
- `ui` - User interface changes
- `data` - Data collection/processing

## 📞 Getting Help

- Check existing [issues](https://github.com/YOUR_USERNAME/fpl-ai-assistant/issues)
- Start a [discussion](https://github.com/YOUR_USERNAME/fpl-ai-assistant/discussions)
- Ask questions in pull request comments

## 🙏 Recognition

Contributors will be recognized in:
- README.md acknowledgments
- Release notes
- Git commit history

Thank you for helping make FPL AI Assistant better! ⚽🚀