
package com.whispercppdemo.ui.main

import android.app.Application
import android.content.Context
import android.media.MediaPlayer
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.net.toUri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.whispercppdemo.media.decodeWaveFile
import com.whispercppdemo.recorder.Recorder
import com.whispercpp.whisper.WhisperContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale

private const val LOG_TAG = "MainScreenViewModel"

class MainScreenViewModel(
    private val application: Application
) : ViewModel() {

    var canTranscribe by mutableStateOf(false)
        private set

    var dataLog by mutableStateOf("")
        private set

    var isRecording by mutableStateOf(false)
        private set

    private val modelsPath = File(application.filesDir, "models")
    private val samplesPath = File(application.filesDir, "samples")

    private val recorder = Recorder()
    private var whisperContext: WhisperContext? = null
    private var mediaPlayer: MediaPlayer? = null
    private var recordedFile: File? = null

    private var tts: TextToSpeech? = null
    @Volatile
    private var ttsReady = false

    // LOCAL TESTING ONLY.
    // Do not commit a real key to GitHub or distribute it inside an APK.
    private val groqApiKey = "Enter API Key"

    init {
        tts = TextToSpeech(application) { status ->
            ttsReady = status == TextToSpeech.SUCCESS

            if (ttsReady) {
                val result = tts?.setLanguage(Locale.US)
                if (result == TextToSpeech.LANG_MISSING_DATA ||
                    result == TextToSpeech.LANG_NOT_SUPPORTED
                ) {
                    Log.w(LOG_TAG, "English TTS voice unavailable")
                }
            }
        }

        viewModelScope.launch {
            printSystemInfo()
            loadData()
        }
    }

    private suspend fun printSystemInfo() {
        printMessage(
            "System Info: ${WhisperContext.getSystemInfo()}\n"
        )
    }

    private suspend fun loadData() {
        printMessage("Loading data...\n")

        try {
            copyAssets()
            loadBaseModel()

            if (whisperContext == null) {
                throw IllegalStateException(
                    "Whisper model was not loaded. Check app/src/main/assets/models."
                )
            }

            canTranscribe = true
            printMessage("Ready. Start recording to speak.\n")
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Model initialization failed", e)
            printMessage("Initialization error: ${e.localizedMessage}\n")
        }
    }

    private suspend fun printMessage(msg: String) =
        withContext(Dispatchers.Main) {
            dataLog += msg
        }

    private suspend fun copyAssets() = withContext(Dispatchers.IO) {
        modelsPath.mkdirs()
        samplesPath.mkdirs()

        // Keep the existing sample-copying behaviour.
        application.copyData("samples", samplesPath) {
            printMessage(it)
        }

        printMessage("Assets copied.\n")
    }

    private suspend fun loadBaseModel() = withContext(Dispatchers.IO) {
        printMessage("Loading Whisper model...\n")

        val models = application.assets.list("models")
            ?.filter { it.endsWith(".bin") }
            .orEmpty()

        if (models.isEmpty()) {
            throw IllegalStateException(
                "No .bin model found in app/src/main/assets/models/"
            )
        }

        // Prefer the tiny multilingual model for this device.
        val modelName = models.firstOrNull {
            it == "ggml-tiny.bin"
        } ?: models.first()

        whisperContext = WhisperContext.createContextFromAsset(
            application.assets,
            "models/$modelName"
        )

        printMessage("Loaded model: $modelName\n")
    }

    // Kept for compatibility with the existing screen.
    // Do not call this: the native benchmark crashes on this device.
    fun benchmark() {
        viewModelScope.launch {
            printMessage(
                "Benchmark disabled because it crashes on this device.\n"
            )
        }
    }

    fun transcribeSample() {
        viewModelScope.launch {
            try {
                val sample = getFirstSample()
                transcribeAudio(sample)
            } catch (e: Exception) {
                Log.e(LOG_TAG, "Sample transcription failed", e)
                printMessage("Sample error: ${e.localizedMessage}\n")
            }
        }
    }

    private suspend fun getFirstSample(): File =
        withContext(Dispatchers.IO) {
            val files = samplesPath.listFiles()
                ?.filter { it.isFile && it.extension == "wav" }
                .orEmpty()

            files.firstOrNull()
                ?: throw IllegalStateException(
                    "No WAV sample found in the samples directory."
                )
        }

    private suspend fun readAudioSamples(file: File): FloatArray =
        withContext(Dispatchers.IO) {
            stopPlayback()
            startPlayback(file)
            decodeWaveFile(file)
        }

    private suspend fun stopPlayback() =
        withContext(Dispatchers.Main) {
            mediaPlayer?.let {
                try {
                    if (it.isPlaying) it.stop()
                } catch (e: Exception) {
                    Log.w(LOG_TAG, "Playback stop failed", e)
                }
                it.release()
            }
            mediaPlayer = null
        }

    private suspend fun startPlayback(file: File) =
        withContext(Dispatchers.Main) {
            mediaPlayer = MediaPlayer.create(
                application,
                file.absolutePath.toUri()
            )
            mediaPlayer?.start()
        }

    private suspend fun transcribeAudio(file: File) {
        if (!canTranscribe) return

        canTranscribe = false

        try {
            printMessage("Reading audio...\n")
            val audioData = readAudioSamples(file)
            printMessage(
                "Audio duration: ${audioData.size / 16} ms\n"
            )

            printMessage("Transcribing locally with Whisper...\n")
            val start = System.currentTimeMillis()

            val transcript = withContext(Dispatchers.Default) {
                whisperContext?.transcribeData(audioData)
            }.orEmpty()

            val elapsed = System.currentTimeMillis() - start

            if (transcript.isBlank()) {
                printMessage("No speech recognized. Please try again.\n")
                return
            }

            printMessage("Transcript ($elapsed ms):\n$transcript\n")
            printMessage("Sending text to AI...\n")

            try {
                val reply = askGroq(transcript)
                printMessage("AI response:\n$reply\n")
                speakReply(reply)
            } catch (e: Exception) {
                Log.e(LOG_TAG, "Groq request failed", e)
                printMessage(
                    "AI request failed: ${e.localizedMessage}\n"
                )
                printMessage(
                    "The local transcript is still available above.\n"
                )
            }
        } catch (e: Exception) {
            Log.e(LOG_TAG, "Transcription failed", e)
            printMessage("Error: ${e.localizedMessage}\n")
        } finally {
            canTranscribe = true
        }
    }

    private suspend fun askGroq(transcript: String): String =
        withContext(Dispatchers.IO) {
            require(
                groqApiKey.isNotBlank() &&
                        groqApiKey != "PASTE_YOUR_GROQ_API_KEY_HERE"
            ) {
                "Set your Groq API key in MainScreenViewModel.kt."
            }

            val connection = URL(
                "https://api.groq.com/openai/v1/chat/completions"
            ).openConnection() as HttpURLConnection

            try {
                connection.requestMethod = "POST"
                connection.connectTimeout = 15000
                connection.readTimeout = 30000
                connection.doOutput = true

                connection.setRequestProperty(
                    "Authorization", "Bearer $groqApiKey"
                )
                connection.setRequestProperty(
                    "Content-Type", "application/json"
                )

                val messages = JSONArray()
                    .put(
                        JSONObject()
                            .put("role", "system")
                            .put(
                                "content",
                                "You are a concise, helpful voice assistant. " +
                                        "Understand English and Hinglish. " +
                                        "Respond naturally in the user's language. " +
                                        "Keep spoken answers brief and clear."
                            )
                    )
                    .put(
                        JSONObject()
                            .put("role", "user")
                            .put("content", transcript)
                    )

                val requestBody = JSONObject()
                    .put("model", "openai/gpt-oss-20b")
                    .put("messages", messages)
                    .put("temperature", 0.3)
                    .put("max_tokens", 250)

                connection.outputStream.use { output ->
                    output.write(
                        requestBody.toString().toByteArray(Charsets.UTF_8)
                    )
                }

                val status = connection.responseCode
                val responseStream =
                    if (status in 200..299) {
                        connection.inputStream
                    } else {
                        connection.errorStream
                    }

                val responseText = responseStream
                    ?.bufferedReader()
                    ?.use { it.readText() }
                    ?: ""

                if (status !in 200..299) {
                    throw IllegalStateException(
                        "Groq returned HTTP $status: $responseText"
                    )
                }

                JSONObject(responseText)
                    .getJSONArray("choices")
                    .getJSONObject(0)
                    .getJSONObject("message")
                    .getString("content")
                    .trim()
            } finally {
                connection.disconnect()
            }
        }

    private suspend fun speakReply(reply: String) =
        withContext(Dispatchers.Main) {
            if (!ttsReady || tts == null) {
                printMessage(
                    "Text-to-speech is not ready; see the text response above.\n"
                )
                return@withContext
            }

            val result = tts?.speak(
                reply,
                TextToSpeech.QUEUE_FLUSH,
                null,
                "assistant_reply"
            )

            if (result == TextToSpeech.ERROR) {
                printMessage("Could not start text-to-speech.\n")
            }
        }

    fun toggleRecord() {
        viewModelScope.launch {
            try {
                if (isRecording) {
                    recorder.stopRecording()
                    isRecording = false

                    recordedFile?.let { file ->
                        transcribeAudio(file)
                    }
                } else {
                    stopPlayback()
                    val file = getTempFileForRecording()

                    recorder.startRecording(file) { error ->
                        viewModelScope.launch {
                            printMessage(
                                "Recording error: ${error.localizedMessage}\n"
                            )
                            isRecording = false
                        }
                    }

                    recordedFile = file
                    isRecording = true
                    printMessage("Recording... tap Stop recording when finished.\n")
                }
            } catch (e: Exception) {
                Log.e(LOG_TAG, "Recording failed", e)
                printMessage("Recording error: ${e.localizedMessage}\n")
                isRecording = false
            }
        }
    }

    private suspend fun getTempFileForRecording(): File =
        withContext(Dispatchers.IO) {
            File.createTempFile("recording", ".wav", application.cacheDir)
        }

    override fun onCleared() {
        // Release native and media resources.
        runBlocking {
            try {
                whisperContext?.release()
                whisperContext = null
                stopPlayback()
            } catch (e: Exception) {
                Log.w(LOG_TAG, "Resource cleanup failed", e)
            }
        }

        tts?.stop()
        tts?.shutdown()
        tts = null
        ttsReady = false

        super.onCleared()
    }

    companion object {
        fun factory() = viewModelFactory {
            initializer {
                val application =
                    this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY]
                            as Application

                MainScreenViewModel(application)
            }
        }
    }
}

private suspend fun Context.copyData(
    assetDirName: String,
    destDir: File,
    printMessage: suspend (String) -> Unit
) = withContext(Dispatchers.IO) {
    assets.list(assetDirName)?.forEach { name ->
        val assetPath = "$assetDirName/$name"
        val destination = File(destDir, name)

        printMessage("Copying $name...\n")

        assets.open(assetPath).use { input ->
            destination.outputStream().use { output ->
                input.copyTo(output)
            }
        }
    }
}
