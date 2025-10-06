import Link from 'next/link';

export default function NotFound() {
  return (
    <div>
      {/* TODO: Style this 404 page when implementing in your project */}
      <h2>Not Found</h2>
      <p>Could not find requested resource</p>
      <Link href="/">Return Home</Link>
    </div>
  );
}