import { faHandPointer } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import React, { useEffect, useRef, useState } from 'react';

import './Chats.scss';

interface ChatsProps {
  userResponse: string;
  botResponse: {
    message: string;
    options?: string[];
  };
  optionClick: (ev: React.MouseEvent<HTMLElement>) => void;
}

interface MessagesInfo {
  message: string;
  options?: string[];
  sender: string;
}

const Chats: React.FC<ChatsProps> = ({ userResponse, botResponse, optionClick }: ChatsProps) => {
  const [messages, setMessages] = useState<MessagesInfo[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);

  // Stack Up Messages:
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([{ message: 'Hi', sender: 'bot' }]);
    } else {
      let tempArray = [...messages];
      tempArray.push({ message: userResponse, sender: 'user' });
      setMessages(tempArray);

      setTimeout(() => {
        let temp2 = [...tempArray];
        temp2.push({ message: botResponse.message, options: botResponse.options, sender: 'bot' });
        setMessages(temp2);
      }, 1000);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [botResponse, userResponse]);

  // Enable Autoscroll After Each Message:
  useEffect(() => {
    if (scrollRef && scrollRef.current && bodyRef && bodyRef.current) {
      bodyRef.current.scrollTo({
        top: scrollRef.current.offsetTop,
        behavior: 'smooth',
      });
    }
  }, [messages]);

  return (
    <div className="message-container" ref={bodyRef}>
      {messages.map((chat) => (
        <div key={chat.message}>
          <div className={`message ${chat.sender}`}>
            <p>{chat.message}</p>
          </div>
          {chat.options ? (
            <div className="options">
              <div>
                <FontAwesomeIcon icon={faHandPointer} />
              </div>
              {chat.options.map((option) => (
                <p onClick={(e) => optionClick(e)} data-id={option} key={option}>
                  {option}
                </p>
              ))}
            </div>
          ) : null}
          <div ref={scrollRef} className="dummy-div"></div>
        </div>
      ))}
    </div>
  );
};

export default Chats;
