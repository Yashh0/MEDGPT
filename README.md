# MedGPT - Medical Knowledge Assistant

A powerful medical knowledge assistant powered by Groq's LLama3-70B model and Streamlit.

## Features

- 🏥 Comprehensive medical information retrieval
- 🔍 Evidence-based responses
- 📚 Structured medical explanations
- 🎯 User-friendly interface
- ⚡ Powered by Groq's LLama3-70B model

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Groq API key:
   - Create a `.streamlit/secrets.toml` file
   - Add your API key:
     ```toml
     [api_keys]
     groq = "your-api-key-here"
     ```
   - This file is gitignored for security

## Local Development

Run the app locally:
```bash
streamlit run app.py
```

## Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set up your API key in Streamlit Cloud:
   - Go to your app settings
   - Find the "Secrets" section
   - Add your API key in this format:
     ```toml
     [api_keys]
     groq = "your-api-key-here"
     ```
5. Deploy!

### Environment Variables

Required environment variables:
- `GROQ_API_KEY`: Your Groq API key

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
