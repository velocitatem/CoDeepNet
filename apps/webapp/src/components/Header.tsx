import Link from "next/link";
import { getLocale } from "@/libs/locales";

export default function Header() {
  const { common } = getLocale('en');

  return (
    <header>
      {/* TODO: Style this header when implementing in your project */}
      <div>
        <div>
          <div>
            <Link href="/">{common.header.brand}</Link>
          </div>
          <nav>
            <Link href="/">{common.header.nav.home}</Link>
            <Link href="/dashboard">{common.header.nav.dashboard}</Link>
            <Link href="/blog">{common.header.nav.blog}</Link>
          </nav>
          <div>
            <Link href="/login">{common.header.actions.signIn}</Link>
          </div>
        </div>
      </div>
    </header>
  );
}
