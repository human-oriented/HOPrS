import type { MetaFunction } from "@remix-run/node";
import { Form, Link } from "@remix-run/react";
import * as Icon from 'react-feather';

export const meta: MetaFunction = () => {
  return [
    { title: "New Remix App" },
    { name: "description", content: "Welcome to Remix!" },
  ];
};

export default function Compare() {
  return (
    <>
    <header className="flex flex-row items-center justify-between w-full shadow-md px-5 absolute top-0 bg-white">
        <img src="/hoprs-logo.png" width={400} />
        <div className="flex flex-row items-center">
            <div className="flex flex-row items-center mx-3 font-bold border-r border-black pr-3">
                <span className="text-xs mr-1">Test User</span>
                <div className="w-8 h-8 rounded-full flex items-center justify-center bg-gray-700 text-white">
                    <Icon.User size={18} />
                </div>
            </div>
            <Icon.Menu size={30} />
        </div>
    </header>
    <div className="p-5 container m-auto text-center items-center justify-center flex flex-col flex-stretch h-screen pt-20">
      <div className="intro text-gray-500 w-2/3 mx-auto mb-5 text-sm">
        <h1 className="font-bold text-3xl text-black">Image Comparison</h1>
        <p className="my-3">Using our tool you can compare an image to a quad tree file. The result of the comparison will show the similaritis between the files and will identify any edits that have been made to the image. </p>
        <p>Please use our <Link className="underline" to="/">encode</Link> tool to generate a quad tree file of any image you'd like to compare.</p>
      </div>
      <Form className="flex flex-col w-full">
        <div className="files flex flex-row items-center">
            <div className="w-1/2 p-5">
                <div className="flex items-center justify-center w-full">
                    <label htmlFor="dropzone-file" className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-gray-800 dark:bg-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-600">
                        <div className="flex flex-col items-center justify-center pt-5 pb-6 text-black">
                            <Icon.FileText size={40} />
                            <p className="my-2 font-bold text-black">Quad tree file upload</p>
                            <p className="mb-2 text-sm text-gray-500 dark:text-gray-400"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                            <p className="text-xs text-gray-500 dark:text-gray-400">.qt file (MAX. 500mb)</p>
                        </div>
                        <input id="dropzone-file" type="file" className="hidden" />
                    </label>
                </div> 
            </div>
            <div className="w-1/2 p-5">
                <div className="flex items-center justify-center w-full">
                        <label htmlFor="dropzone-file" className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-gray-800 dark:bg-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-600">
                            <div className="flex flex-col items-center justify-center pt-5 pb-6 text-black">
                                <Icon.Image size={40} />
                                <p className="my-2 font-bold text-black">Comparison image upload</p>
                                <p className="mb-2 text-sm text-gray-500 dark:text-gray-400"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                                <p className="text-xs text-gray-500 dark:text-gray-400">HEIC, PNG, JPG or GIF (MAX. 4000x3000px)</p>
                            </div>
                            <input id="dropzone-file" type="file" className="hidden" />
                        </label>
                    </div> 
                </div>
        </div>
        
        <div className="options flex flex-col text-left items-center px-5">
            <div className="heading flex flex-row items-center">
                <button className="font-bold text-sm" type="button">Options</button>
                <Icon.ChevronDown size={15} />
            </div>
            
            <div className="options-inner hidden">
                <div className="flex flex-col w-1/2 text-left justify-start">
                    <label>Threshold</label>
                    <input type="number" name="threshold" value={2} />
                </div>
                <div className="flex flex-col w-1/2 items-center">
                    <label>Depth</label>
                    <input type="number" name="depth" value={5} />
                </div>
            </div>
        </div>
        <div className="buttons my-10">
            <button className="bg-black hover:opacity-90 text-white font-bold px-10 py-4 rounded-tl-2xl rounded-br-2xl" style={{minWidth: 300}}>Compare Image</button>
        </div>
      </Form>
    </div>
    </>
  );
}
