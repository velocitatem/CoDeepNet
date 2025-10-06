import { getLocale } from "@/libs/locales";

export default function FeaturesGrid() {
  const { common } = getLocale('en');

  return (
    <section>
      {/* TODO: Style this features grid when implementing in your project */}
      <div>
        <h2>{common.features.title}</h2>
        <div>
          {common.features.items.map((feature, index) => (
            <div key={index}>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}