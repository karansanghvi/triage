import { useEffect, useState } from "react";

const NAME_KEY = "triage.visitorName"
const ASKED_KEY = "triage.nameAsked"
const CHANGE_EVENT = "triage:name-changed"

function readName(): string {
    return localStorage.getItem(NAME_KEY) ?? ""
}

export function useVisitorName() {
    const [name, setNameState] = useState<string>(readName)
    const [asked, setAsked] = useState<boolean>(() => localStorage.getItem(ASKED_KEY) === "true")

    // Keep every component in sync when the name changes anywhere 
    useEffect(() => {
        const sync = () => setNameState(readName())
        window.addEventListener(CHANGE_EVENT, sync)
        return () => window.removeEventListener(CHANGE_EVENT, sync)
    }, []);

    function markAsked() {
        localStorage.setItem(ASKED_KEY, "true")
        setAsked(true)
    }

    function setName(newName: string) {
        const trimmed = newName.trim()
        if (trimmed) localStorage.setItem(NAME_KEY, trimmed)
        else localStorage.removeItem(NAME_KEY)
        markAsked()
        window.dispatchEvent(new Event(CHANGE_EVENT))
    }

    return { name, setName, asked, skip: markAsked }
}