// One chat request at a time, decided synchronously. React state (the chat's `busy`) only changes on the next
// render, so two sends in the same moment (Enter and a click, a double-click, a double-tap on Try again) would
// both see busy = false and send the same message twice, each using one of the free demo's requests. The chat
// keeps one guard in a ref: run() claims it before anything else happens, and frees it when the request ends,
// however it ends. Pure logic, no React: tests run it directly.

export type SendGuard = {
  busy: () => boolean
  // Starts task() unless one is already running. Returns its promise, or null if it was refused (nothing ran).
  run: <T>(task: () => Promise<T>) => Promise<T> | null
}

export function createSendGuard(): SendGuard {
  let inFlight = false
  const release = () => {
    inFlight = false
  }
  return {
    busy: () => inFlight,
    run<T>(task: () => Promise<T>): Promise<T> | null {
      if (inFlight) return null
      inFlight = true
      let started: Promise<T>
      try {
        started = task()
      } catch (err) {
        release()
        throw err
      }
      return started.finally(release)
    },
  }
}
