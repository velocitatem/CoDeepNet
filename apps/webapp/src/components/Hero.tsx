import Link from "next/link";
import { getLocale } from "@/libs/locales";

export default function Hero() {
  const { common } = getLocale('en');
  
  return (
    <section>
      {/* TODO: Style this hero section when implementing in your project */}
      <div>
        <h1>{common.hero.title}</h1>
        <p>{common.hero.description}</p>
        <div>
          <Link href="/dashboard">
            <button>{common.hero.actions.getStarted}</button>
          </Link>
          <Link href="/blog">
            <button>{common.hero.actions.learnMore}</button>
          </Link>
        </div>
      </div>
    </section>
  );
}