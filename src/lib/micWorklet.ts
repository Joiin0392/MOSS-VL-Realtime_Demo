// Mic capture: MediaStream → AudioWorklet (public/worklets/pcm-worklet.js) →
// 160 ms Int16 PCM chunks @16 kHz for the session socket's 0x01 lane.
// AudioWorklet, not the deprecated ScriptProcessor (backend_overhaul.md §F2).

export interface MicCapture {
  /** Gate chunk delivery without touching the MediaStream (PTT / mute). */
  setForwarding(on: boolean): void;
  stop(): Promise<void>;
}

export async function startMicCapture(
  stream: MediaStream,
  onChunk: (pcm: ArrayBuffer) => void,
): Promise<MicCapture> {
  if (stream.getAudioTracks().length === 0) {
    throw new Error('media stream has no audio track');
  }
  const ctx = new AudioContext();
  // BASE_URL is './' behind the gateway — the worklet must resolve relative to
  // the page, never the domain root (same rule as every other asset).
  await ctx.audioWorklet.addModule(`${import.meta.env.BASE_URL}worklets/pcm-worklet.js`);
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, 'pcm-worklet', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    channelCount: 1,
    channelCountMode: 'explicit',
  });

  let forwarding = true;
  node.port.onmessage = (msg: MessageEvent<ArrayBuffer>) => {
    if (forwarding) onChunk(msg.data);
  };

  // A zero-gain sink keeps the node alive in the graph without audible output.
  const sink = ctx.createGain();
  sink.gain.value = 0;
  source.connect(node);
  node.connect(sink);
  sink.connect(ctx.destination);
  if (ctx.state === 'suspended') await ctx.resume();

  return {
    setForwarding(on: boolean) {
      forwarding = on;
    },
    async stop() {
      forwarding = false;
      node.port.onmessage = null;
      try {
        source.disconnect();
        node.disconnect();
        sink.disconnect();
      } catch {
        /* graph may already be torn down */
      }
      await ctx.close().catch(() => undefined);
    },
  };
}
