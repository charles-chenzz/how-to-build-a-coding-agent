package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
)

func main() {
	verbose := flag.Bool("verbose", false, "enable verbose logging")
	flag.Parse()

	if *verbose {
		log.SetOutput(os.Stderr)
		log.SetFlags(log.LstdFlags | log.Lshortfile)
		log.Println("Verbose logging enabled")
	} else {
		log.SetOutput(os.Stdout)
		log.SetFlags(0)
		log.SetPrefix("")
	}

	client := anthropic.NewClient()
	if *verbose {
		log.Println("Anthropic client initialized")
	}

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	agent := NewAgent(&client, getUserMessage, *verbose)
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

func NewAgent(client *anthropic.Client, getUserMessage func() (string, bool), verbose bool) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		verbose:        verbose,
	}
}

type Agent struct {
	client         *anthropic.Client
	getUserMessage func() (string, bool)
	verbose        bool
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []anthropic.MessageParam{}

	if a.verbose {
		log.Println("Starting chat session")
	}
	fmt.Println("Chat with Claude (use 'ctrl-c' to quit)")

	for {
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			if a.verbose {
				log.Println("User input ended, breaking from chat loop")
			}
			break
		}

		// Skip empty messages
		if userInput == "" {
			if a.verbose {
				log.Println("Skipping empty message")
			}
			continue
		}

		// Handle special commands
		if userInput == "/history" {
			if len(conversation) == 0 {
				fmt.Println("对话历史为空")
			} else {
				fmt.Printf("对话历史 (%d 条消息):\n", len(conversation))
				for i, conv := range conversation {
					fmt.Printf("  [%d] %+v\n", i+1, conv)
				}
			}
			continue  // 不发送给 Claude
		}

		if userInput == "/clear" {
			oldLen := len(conversation)
			conversation = []anthropic.MessageParam{}
			fmt.Printf("已清空对话历史 (删除了 %d 条消息)\n", oldLen)
			continue  // 不发送给 Claude
		}

		if userInput == "/help" {
			fmt.Println("可用命令:")
			fmt.Println("  /history  - 显示对话历史")
			fmt.Println("  /clear    - 清空对话历史")
			fmt.Println("  /help     - 显示帮助信息")
			continue  // 不发送给 Claude
		}

		if a.verbose {
			log.Printf("User input received: %q", userInput)
		}

		userMessage := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
		conversation = append(conversation, userMessage)

		if a.verbose {
			log.Printf("Sending message to Claude, conversation length: %d", len(conversation))
		}

		message, err := a.runInference(ctx, conversation)
		if err != nil {
			if a.verbose {
				log.Printf("Error during inference: %v", err)
			}
			return err
		}
		conversation = append(conversation, message.ToParam())

		if a.verbose {
			log.Printf("Received response from Claude with %d content blocks", len(message.Content))
		}

		for _, content := range message.Content {
			switch content.Type {
			case "text":
				fmt.Printf("\u001b[93mClaude\u001b[0m: %s\n", content.Text)
			}
		}
	}

	if a.verbose {
		log.Println("Chat session ended")
	}
	return nil
}

func (a *Agent) runInference(ctx context.Context, conversation []anthropic.MessageParam) (*anthropic.Message, error) {
	if a.verbose {
		log.Printf("Making API call to Claude with model: %s", anthropic.ModelClaude3_7SonnetLatest)
	}

	message, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaude3_7SonnetLatest,
		MaxTokens: int64(1024),
		Messages:  conversation,
	})

	if a.verbose {
		if err != nil {
			log.Printf("API call failed: %v", err)
		} else {
			log.Printf("API call successful, response received")
		}
	}

	return message, err
}
