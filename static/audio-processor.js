/**
 * AudioWorklet processor for capturing audio from the microphone
 * Sends audio chunks to the main thread for transmission to the server
 */
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 2048;  // Accumulate samples before sending
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];

        if (input && input.length > 0) {
            const channelData = input[0];

            // Accumulate samples into buffer
            for (let i = 0; i < channelData.length; i++) {
                this.buffer[this.bufferIndex] = channelData[i];
                this.bufferIndex++;

                // When buffer is full, send to main thread
                if (this.bufferIndex >= this.bufferSize) {
                    // Copy buffer and send
                    this.port.postMessage(this.buffer.slice(0, this.bufferIndex));
                    this.bufferIndex = 0;
                }
            }
        }

        return true;  // Keep processor alive
    }
}

registerProcessor('audio-processor', AudioProcessor);
