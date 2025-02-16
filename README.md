# Hello World Agent

A simple demonstration agent using the ReACT methodology for analyzing and executing tasks.

## Quick Install

```bash
pip install hello_agent
```

## Prerequisites

- Python 3.8 or higher
- OpenRouter API key (for LLM access)

## Installation

1. Install the package:
   ```bash
   pip install hello_agent
   ```

2. Add your OpenRouter API key to the `.env` file:
   ```bash
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Usage

The agent can be run in two ways:

1. Using the command-line tool:
   ```bash
   agent --prompt "What is quantum computing?" --task research
   ```

2. Using Python code:
   ```python
   from agent.crew import HelloWorldCrew
   
   crew = HelloWorldCrew()
   result = crew.run(prompt="What is quantum computing?", task_type="research")
   ```

### Command Line Arguments

- `--prompt`: Specify the input prompt (default: "Tell me about yourself")
- `--task`: Specify the task type: research, execute, analyze, or both (default: both)
- `--hitl`: Enable human-in-the-loop mode (optional)

Example:
```bash
agent --prompt "What is quantum computing?" --task research --hitl
```

## Features

- ReACT Methodology Implementation
- Research Analysis
- Task Execution
- Performance Analysis
- Progress Tracking
- Streaming Responses
- Optional Human-in-the-Loop Mode

## Documentation

For detailed documentation and user guides, refer to:

- [User Guide](agent/docs/readme.md)
- [Templates Guide](agent/docs/templates.md)
- [Tools Guide](agent/docs/tools.md)
- [Configuration Guide](agent/docs/configuration.md)
- [Advanced Implementations Guide](agent/docs/advanced_implementations.md)
- [Memory and Storage Guide](agent/docs/memory_and_storage.md)
- [Human-in-the-Loop Guide](agent/docs/human_in_the_loop.md)

## Examples

Explore the [Examples](agent/examples/README.md) directory for sample usage scenarios and human-in-the-loop implementations.

## Project Structure

```
hello_world/
├── config/              # Configuration files
│   ├── agents.yaml     # Agent definitions
│   ├── tasks.yaml      # Task definitions
│   └── analysis.yaml   # Analysis rules
├── tools/              # Custom tools
├── docs/               # Documentation
└── examples/           # Example implementations
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [CrewAI](https://github.com/joaomdmoura/crewAI)
- Powered by [OpenRouter](https://openrouter.ai/)
