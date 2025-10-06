class PCMWorkletProcessor extends AudioWorkletProcessor {
  process(inputs) {
    if (!inputs || inputs.length === 0) {
      return true;
    }

    const channelData = inputs[0];
    if (!channelData || channelData.length === 0) {
      return true;
    }

    const samples = channelData[0];
    if (!samples || samples.length === 0) {
      return true;
    }

    // Copy samples into transferable buffer to avoid blocking audio thread
    const copy = new Float32Array(samples.length);
    copy.set(samples);

    this.port.postMessage(
      {
        samples: copy,
        sampleRate,
      },
      [copy.buffer]
    );

    return true;
  }
}

registerProcessor('pcm-worklet-processor', PCMWorkletProcessor);
