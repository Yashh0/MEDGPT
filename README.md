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
   - Create a `.env` file in the root directory
   - Add your API key: `GROQ_API_KEY=your_api_key_here`

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
4. Add your Groq API key in the Streamlit Cloud secrets management
5. Deploy!

### Environment Variables

Required environment variables:
- `GROQ_API_KEY`: Your Groq API key

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
