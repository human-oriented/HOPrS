import type { MetaFunction } from "@remix-run/node";
import { Link } from "@remix-run/react";

export const meta: MetaFunction = () => {
  return [
    { title: "New Remix App" },
    { name: "description", content: "Welcome to Remix!" },
  ];
};

export default function Index() {
  return (
    <div className="text-2xl font-bold p-5 container m-auto text-center items-center flex flex-col">
      <img src="/hoprs-logo.png" width={500} />
      <Link to="/compare">Compare</Link>
      <Link to="/compare">Encode</Link>
    </div>
  );
}
