// The browser exposes document rendering feedback, with different coverage for each API.
export function observeBrowserFrameFeedback(target) {
  const Observer = target.PerformanceObserver;
  const entries = [];
  const types = ['event', 'long-animation-frame'].filter(type => Observer?.supportedEntryTypes?.includes(type));
  const observers = types.map(type => {
    const observer = new Observer(list => entries.push(...list.getEntries()));
    observer.observe(type === 'event' ? { type, durationThreshold: 16 } : { type });
    return observer;
  });
  return {
    types,
    stop() {
      for (const observer of observers) {
        entries.push(...observer.takeRecords());
        observer.disconnect();
      }
      return entries;
    },
  };
}

const positiveTime = time => Number.isFinite(time) && time > 0 ? time : null;
const only = values => values.length === 1 ? values[0] : null;

export function browserFrameFeedback(frame, event, entries) {
  const { callbackStartMs, previousRenderEndMs, nextCallbackStartMs, presentReturnedMs } = frame;
  const key = event?.trigger === 'keyboard'
    ? only(entries.filter(entry => entry.entryType === 'event' && entry.name === 'keydown'
      && entry.startTime === event.eventMs
      && entry.processingStart <= event.acceptedMs && entry.processingEnd >= event.acceptedMs))
    : null;

  // Use the rendering-cycle start to find the document frame that ran this callback.
  // Presentation may arrive later. A different document frame is never a substitute.
  const documentFrame = presentReturnedMs === null ? null : only(entries.filter(entry =>
    entry.entryType === 'long-animation-frame' && entry.renderStart >= previousRenderEndMs
    && entry.renderStart <= callbackStartMs && entry.renderStart > 0
    && (!entry.styleAndLayoutStart || entry.styleAndLayoutStart >= presentReturnedMs)
    && (!entry.paintTime || entry.paintTime >= presentReturnedMs)));
  const presentationTime = positiveTime(documentFrame?.presentationTime);
  const documentPresentationMs = presentationTime !== null && presentationTime >= presentReturnedMs
    ? presentationTime : null;
  const presentation = [];
  if (documentPresentationMs !== null) {
    presentation.push({ source: 'long-animation-frame', kind: 'browser-feedback', atMs: documentPresentationMs });
  }
  if (nextCallbackStartMs !== null && presentReturnedMs !== null && nextCallbackStartMs >= presentReturnedMs) {
    presentation.push({ source: 'next-animation-frame', kind: 'estimate', atMs: nextCallbackStartMs });
  }

  return {
    presentation,
    documentFrame: documentFrame === null ? null : {
      renderStartMs: documentFrame.renderStart,
      paintMs: positiveTime(documentFrame.paintTime),
      presentationMs: documentPresentationMs,
    },
    documentKey: key === null ? null : {
      processingStartMs: key.processingStart,
      processingEndMs: key.processingEnd,
      durationMs: key.duration,
      interactionId: key.interactionId ?? null,
      paintMs: positiveTime(key.paintTime),
      presentationMs: positiveTime(key.presentationTime),
    },
  };
}
