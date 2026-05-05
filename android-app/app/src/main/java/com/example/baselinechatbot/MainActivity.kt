//Entrypoint tell android which screen to show fist

package com.example.baselinechatbot

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.material3.Surface
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.baselinechatbot.ui.theme.BaselineChatbotTheme
import androidx.compose.ui.graphics.Color

// Creates the main android screen container and is base class for activities

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            BaselineChatbotTheme {
                Surface(color = Color(0xFFF5F7FB)) {
                    ChatScreen(viewModel = viewModel())
                }
            }
        }
    }
}

