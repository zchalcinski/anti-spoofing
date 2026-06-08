package com.chalksoftware.voicespoofdetector

import android.content.Context
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.LiteModuleLoader
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

import com.topjohnwu.superuser.Shell // Import libsu

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.core.app.ActivityCompat
import android.Manifest
import android.content.pm.PackageManager

import android.media.AudioManager

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val statusText = findViewById<TextView>(R.id.statusText)
        val testButton = findViewById<Button>(R.id.testButton)
        val rootButton = findViewById<Button>(R.id.rootButton)
        val callButton = findViewById<Button>(R.id.callButton)

        testButton.setOnClickListener {
            statusText.text = "Loading model..."

            // Run this in a background thread so the UI doesn't freeze
            Thread {
                try {
                    // 1. Prepare the file (copy from assets to cache)
                    // MAKE SURE YOUR FILE IS NAMED "model.ptl" IN ASSETS
                    val modelPath = "/data/local/tmp/model.ptl"

                    // 2. Load the model
                    val module = LiteModuleLoader.load(modelPath)

                    // 3. Update UI (must go back to main thread)
                    runOnUiThread {
                        statusText.text = "Success! Model loaded.\nPath: $modelPath"
                    }

                    Log.d("SpoofDetect", "Model loaded successfully")

                } catch (e: Exception) {
                    Log.e("SpoofDetect", "Error loading model", e)
                    runOnUiThread {
                        statusText.text = "Error: ${e.message}"
                    }
                }
            }.start()
        }

        rootButton.setOnClickListener {
            statusText.text = "Requesting Root..."

            // This runs in background automatically by libsu
            Shell.getShell { shell ->
                if (shell.isRoot) {
                    // WE HAVE POWER!
                    runOnUiThread {
                        statusText.text = "Root Granted! ready to capture."
                        Log.d("SpoofDetect", "Root access granted.")
                    }

                    // TODO: Start the actual recording here

                } else {
                    runOnUiThread {
                        statusText.text = "Root Denied. Cannot proceed."
                        Log.e("SpoofDetect", "Root access denied.")
                    }
                }
            }
        }

        callButton.setOnClickListener {
            startCallRecording()
        }
    }

    val PERMISSION_CODE = 200

    fun startCallRecording() {
        // 1. Check standard permissions
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.READ_EXTERNAL_STORAGE), PERMISSION_CODE)
            return
        }

        Thread {
            try {
                // 2. Configure the recorder for the FORBIDDEN source
                // AudioSource.VOICE_CALL (4) captures the actual phone call (uplink + downlink)
                val audioSource = MediaRecorder.AudioSource.VOICE_COMMUNICATION
                val sampleRate = 16000 // Phone calls are 16kHz
                val channelConfig = AudioFormat.CHANNEL_IN_MONO
                val audioFormat = AudioFormat.ENCODING_PCM_16BIT

                val minBufSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)

                // 3. Initialize Recorder
                val recorder = AudioRecord(audioSource, sampleRate, channelConfig, audioFormat, minBufSize)

                if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                    Log.e("SpoofDetect", "Recorder failed to initialize! (Are we a system app yet?)")
                    return@Thread
                }

                val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager

                // FORCE the device to know it's in a call
                audioManager.mode = AudioManager.MODE_IN_COMMUNICATION

                recorder.startRecording()
                Log.d("SpoofDetect", "Recording STARTED from VOICE_CALL source!")

                // 4. Read data and Save to file for testing
                val buffer = ShortArray(minBufSize)
                val outputFile = File(filesDir, "call_recording_test.pcm")
                val fos = FileOutputStream(outputFile)

                // Record for 10 seconds (approx)
                val startTime = System.currentTimeMillis()
                while (System.currentTimeMillis() - startTime < 10000) {
                    val readCount = recorder.read(buffer, 0, minBufSize)
                    if (readCount > 0) {
                        // We have raw audio data!
                        // Convert ShortArray to Byte array for writing to file
                        val bytes = ByteArray(readCount * 2)
                        java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(buffer, 0, readCount)
                        fos.write(bytes)

                        // TODO: LATER -> Send 'buffer' to your PyTorch model here
                    }
                }

                recorder.stop()
                recorder.release()
                fos.close()
                Log.d("SpoofDetect", "Recording finished. Saved to ${outputFile.absolutePath}")

            } catch (e: Exception) {
                Log.e("SpoofDetect", "Crash: ${e.message}")
                e.printStackTrace()
            }
        }.start()
    }

    // --- Helper Function (Boilerplate) ---
    // Android can't read files directly from the "assets" folder into C++.
    // We must copy it to a real file path first.
    @Throws(IOException::class)
    fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }
}