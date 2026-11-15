type UserMessageProps = {
    content: string
}

export default function UserMessage({ content }: UserMessageProps) {
  return (
    <>
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-primary/15 px-4 py-3 text-sm text-foreground">
            {content}
        </div>
      </div>
    </>
  )
}
