'use client';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div>
      {/* TODO: Style this error page when implementing in your project */}
      <h2>Something went wrong!</h2>
      <p>{error.message || 'An unexpected error occurred'}</p>
      <button onClick={() => reset()}>
        Try again
      </button>
    </div>
  );
}