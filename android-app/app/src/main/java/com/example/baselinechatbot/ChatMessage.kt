// Defines what one message looks like

package com.example.baselinechatbot

//One chat message has two pieces of information:
// string text : stores the content of the message.
// isUser : tells the app who sent the message.
//true → message came from the user
//false → message came from the chatbot
// UI uses it to decide where to place the message bubble:
//user messages go to the right and bot messages go to the left
data class ChatMessage(
    val text: String,
    val isUser: Boolean,
    val timestamp : String
)