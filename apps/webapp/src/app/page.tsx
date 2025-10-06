import Hero from "@/components/Hero";
import FeaturesGrid from "@/components/FeaturesGrid";
import Testimonials1 from "@/components/Testimonials1";
import Pricing from "@/components/Pricing";
//import FAQ from "@/components/FAQ";
//import CTA from "@/components/CTA";

export default function Home() {
  return (
    <div>
      <Hero />
      <FeaturesGrid />
      <Testimonials1 />
      <Pricing />
    </div>
  );
}
